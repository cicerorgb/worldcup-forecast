"""
2026 FIFA World Cup — Score Forecaster (Round 3 of the Group Stage)

Pipeline:
  1. Load data  (FIFA rankings + schedule CSV + per-team HTML match history)
  2. Feature engineering with exponential time-decay over history (2020–2026)
  3. LOO-CV on 3 models with and without historical data (baseline comparison)
  4. Inverse-MAE weighted ensemble
  5. Estimate Dixon-Coles ρ parameter
  6. Generate predictions: expected goals (λ), W/D/L probabilities, most likely score
  7. Save predictions CSV and charts to outputs/
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

warnings.filterwarnings("ignore")

from src.data_loader import load_rankings, load_matches
from src.html_parser import load_historical
from src.features import build_training_data, build_prediction_features, FEATURE_COLS
from src.models.poisson_model import (
    build_model as build_poisson,
    estimate_rho,
    most_likely_score,
    result_probabilities,
    score_probability_matrix,
)
from src.models.boosting import build_xgb, build_lgb, WeightedEnsemble
from src.evaluation import loo_cv

OUTPUT_DIR = Path("outputs")
CHARTS_DIR = OUTPUT_DIR / "charts"
OUTPUT_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _result_label(g1: int, g2: int, t1: str, t2: str) -> str:
    if g1 > g2:
        return f"{t1} Win"
    if g1 < g2:
        return f"{t2} Win"
    return "Draw"


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(xgb_model, path: Path) -> None:
    try:
        reg = xgb_model.named_steps["reg"] if hasattr(xgb_model, "named_steps") else xgb_model
        importances = reg.feature_importances_
    except AttributeError:
        importances = xgb_model.feature_importances_

    fi = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True)
    labels = {
        "attack_off":    "Attack rate (attacking team)",
        "defense_def":   "Defence rate (opposing team)",
        "rank_off":      "FIFA ranking (attacking team)",
        "rank_def":      "FIFA ranking (opposing team)",
        "pts_off":       "Points per game (attacking team)",
        "pts_def":       "Points per game (opposing team)",
        "rank_diff":     "Ranking difference",
        "atk_vs_def":    "Attack vs Defence",
        "quality_ratio": "Quality ratio",
    }
    fi.index = [labels.get(c, c) for c in fi.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1a73e8" if v >= fi.median() else "#90caf9" for v in fi.values]
    fi.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Importance (gain)", fontsize=11)
    ax.set_title("Feature Importance — XGBoost (with historical data)", fontsize=13, fontweight="bold")
    ax.axvline(fi.median(), color="red", linestyle="--", alpha=0.6, label="Median")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_score_heatmaps(predictions: list[dict], rho: float, path: Path) -> None:
    # Team names match source data (Portuguese canonical names)
    highlights = ["Brasil", "Argentina", "Espanha", "França", "Inglaterra", "Portugal"]
    selected = [p for p in predictions if p["team1"] in highlights or p["team2"] in highlights][:6]
    if not selected:
        selected = predictions[:6]

    ncols = 3
    nrows = -(-len(selected) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4.5))
    axes = np.array(axes).flatten()

    for i, pred in enumerate(selected):
        ax = axes[i]
        l1, l2 = pred["lambda1"], pred["lambda2"]
        matrix = score_probability_matrix(l1, l2, rho, max_goals=6)[:7, :7]
        sns.heatmap(matrix * 100, annot=True, fmt=".1f", cmap="Blues",
                    ax=ax, cbar=False, linewidths=0.5)
        ax.set_xlabel(pred["team2"], fontsize=10, fontweight="bold")
        ax.set_ylabel(pred["team1"], fontsize=10, fontweight="bold")
        ax.set_title(
            f"Group {pred['group']} | {pred['team1']} vs {pred['team2']}\n"
            f"Most likely score: {pred['g1']}–{pred['g2']}",
            fontsize=9, fontweight="bold",
        )

    for j in range(len(selected), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Score Probability Matrix (%) — Round 3",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_result_probs(predictions: list[dict], path: Path) -> None:
    n = len(predictions)
    fig, ax = plt.subplots(figsize=(14, n * 0.55 + 1.5))

    labels = [f"G{p['group']}: {p['team1']} vs {p['team2']}" for p in predictions]
    pw  = [p["p_win1"] for p in predictions]
    pd_ = [p["p_draw"] for p in predictions]
    pl  = [p["p_win2"] for p in predictions]
    y   = np.arange(n)
    h   = 0.6

    ax.barh(y, pw,       height=h, color="#1a73e8", label="Team 1 Win")
    ax.barh(y, pd_, left=pw, height=h, color="#e0e0e0", label="Draw")
    ax.barh(y, pl,  left=[a + b for a, b in zip(pw, pd_)], height=h,
            color="#e53935", label="Team 2 Win")

    for i, (p1, pe, p2) in enumerate(zip(pw, pd_, pl)):
        if p1 > 0.12:
            ax.text(p1 / 2, i, f"{p1:.0%}", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")
        if pe > 0.08:
            ax.text(p1 + pe / 2, i, f"{pe:.0%}", ha="center", va="center",
                    fontsize=7.5, color="#444")
        if p2 > 0.12:
            ax.text(p1 + pe + p2 / 2, i, f"{p2:.0%}", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", fontsize=10)
    ax.set_title("Win / Draw / Loss Probabilities — Round 3 (with historical data)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(
    baseline: dict[str, dict],
    with_hist: dict[str, dict],
    path: Path,
) -> None:
    """Bar chart comparing MAE and result accuracy with vs. without historical data."""
    models = list(baseline.keys())
    x = np.arange(len(models))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    mae_base = [baseline[m]["mae"] for m in models]
    mae_hist = [with_hist[m]["mae"] for m in models]
    acc_base = [baseline[m]["result_acc_pct"] for m in models]
    acc_hist = [with_hist[m]["result_acc_pct"] for m in models]

    ax1.bar(x - w / 2, mae_base, w, label="No history", color="#90caf9")
    ax1.bar(x + w / 2, mae_hist, w, label="With history", color="#1a73e8")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylabel("MAE (goals)", fontsize=10)
    ax1.set_title("Mean Absolute Error (lower is better)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    for xi, (b, h) in enumerate(zip(mae_base, mae_hist)):
        delta = b - h
        color = "green" if delta > 0 else "red"
        ax1.text(xi + w / 2, h + 0.01, f"{delta:+.3f}", ha="center", va="bottom",
                 fontsize=8, color=color)

    ax2.bar(x - w / 2, acc_base, w, label="No history", color="#ef9a9a")
    ax2.bar(x + w / 2, acc_hist, w, label="With history", color="#e53935")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.set_ylabel("Result Accuracy (%)", fontsize=10)
    ax2.set_title("W/D/L Accuracy (higher is better)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    for xi, (b, h) in enumerate(zip(acc_base, acc_hist)):
        delta = h - b
        color = "green" if delta > 0 else "red"
        ax2.text(xi + w / 2, h + 0.3, f"{delta:+.1f}%", ha="center", va="bottom",
                 fontsize=8, color=color)

    fig.suptitle("Impact of Historical Data (2020–2026) on Model Performance (LOO-CV)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  2026 FIFA WORLD CUP — SCORE FORECASTER (ROUND 3)")
    print("=" * 65)

    # 1. DATA
    print("\n[1/6] Loading data...")
    rankings = load_rankings()
    played, future = load_matches()

    historical_raw = load_historical(year_from=2020)
    # Exclude 2026 World Cup matches from history (handled separately via played)
    historical = historical_raw[
        ~historical_raw["tournament"].str.contains("Copa do Mundo da FIFA 2026", na=False)
    ].copy()

    games_per_team = historical.groupby("team").size()
    print(f"      {len(played)} matches played | {len(future)} to predict")
    print(f"      {len(rankings)} teams with FIFA ranking")
    print(f"      {len(historical)} historical records ({historical['date'].dt.year.min()}–2026)")
    print(f"      Average of {games_per_team.mean():.0f} matches per team in historical data")

    builders = {
        "Poisson GLM": build_poisson,
        "XGBoost":     build_xgb,
        "LightGBM":    build_lgb,
    }

    # 2. LOO-CV WITHOUT HISTORY (baseline)
    print("\n[2/6] LOO-CV without historical data (baseline)...")
    cv_base: dict[str, dict] = {}
    for name, builder in builders.items():
        print(f"      -> {name}...", end=" ", flush=True)
        cv = loo_cv(played, rankings, builder, historical=None)
        cv_base[name] = cv
        print(f"MAE={cv['mae']:.3f} | Result acc={cv['result_acc_pct']:.1f}% | Exact score={cv['exact_score_pct']:.1f}%")

    # 3. LOO-CV WITH HISTORY
    print("\n[3/6] LOO-CV with historical data (2020–2026)...")
    cv_hist: dict[str, dict] = {}
    for name, builder in builders.items():
        print(f"      -> {name}...", end=" ", flush=True)
        cv = loo_cv(played, rankings, builder, historical=historical)
        cv_hist[name] = cv
        print(f"MAE={cv['mae']:.3f} | Result acc={cv['result_acc_pct']:.1f}% | Exact score={cv['exact_score_pct']:.1f}%")

    print("\n      Gain from historical data:")
    for name in builders:
        delta_mae = cv_base[name]["mae"] - cv_hist[name]["mae"]
        delta_acc = cv_hist[name]["result_acc_pct"] - cv_base[name]["result_acc_pct"]
        sign_mae = "improvement" if delta_mae > 0 else "degradation"
        sign_acc = "improvement" if delta_acc > 0 else "degradation"
        print(f"        {name}: MAE {delta_mae:+.3f} ({sign_mae}) | "
              f"Result acc {delta_acc:+.1f}% ({sign_acc})")

    # 4. ENSEMBLE WEIGHTS (inverse MAE from LOO-CV with history)
    maes = {n: cv_hist[n]["mae"] for n in builders}
    inv_sum = sum(1 / v for v in maes.values())
    weights = [1 / maes[n] / inv_sum for n in builders]
    print(f"\n      Ensemble weights: " +
          " | ".join(f"{n}={w:.2f}" for n, w in zip(builders.keys(), weights)))

    # 5. FINAL MODELS
    print("\n[4/6] Training final models on all 48 matches + historical data...")
    X_train, y_train = build_training_data(played, rankings, historical=historical)

    poisson_final = build_poisson()
    xgb_final     = build_xgb()
    lgb_final     = build_lgb()
    poisson_final.fit(X_train, y_train)
    xgb_final.fit(X_train, y_train)
    lgb_final.fit(X_train, y_train)

    ensemble = WeightedEnsemble([poisson_final, xgb_final, lgb_final], weights)
    ensemble.fit(X_train, y_train)

    rho = estimate_rho(played, ensemble, X_train)
    print(f"      Estimated Dixon-Coles ρ: {rho:.4f}")

    # 6. PREDICTIONS
    print("\n[5/6] Generating Round 3 predictions...")
    X_fut1, X_fut2 = build_prediction_features(future, played, rankings, historical=historical)
    lambda1_all = np.clip(ensemble.predict(X_fut1), 0.05, None)
    lambda2_all = np.clip(ensemble.predict(X_fut2), 0.05, None)

    predictions = []
    for i, (_, row) in enumerate(future.iterrows()):
        l1, l2 = float(lambda1_all[i]), float(lambda2_all[i])
        g1, g2, prob = most_likely_score(l1, l2, rho)
        p_win1, p_draw, p_win2 = result_probabilities(l1, l2, rho)
        predictions.append({
            "date":       row["date"].strftime("%d/%m"),
            "group":      row["group"],
            "team1":      row["team1"],
            "team2":      row["team2"],
            "g1": g1, "g2": g2,
            "lambda1":    round(l1, 3),
            "lambda2":    round(l2, 3),
            "p_win1":     p_win1,
            "p_draw":     p_draw,
            "p_win2":     p_win2,
            "score_prob": prob,
        })

    print()
    table_rows = [[
        p["date"], p["group"], p["team1"],
        f"{p['g1']}–{p['g2']}",
        p["team2"],
        f"λ {p['lambda1']:.2f}×{p['lambda2']:.2f}",
        f"{p['p_win1']:.0%} / {p['p_draw']:.0%} / {p['p_win2']:.0%}",
        f"{p['score_prob']:.1%}",
        _result_label(p["g1"], p["g2"], p["team1"], p["team2"]),
    ] for p in predictions]

    print(tabulate(
        table_rows,
        headers=["Date", "Grp", "Team 1", "Score", "Team 2",
                 "Exp. Goals", "W1/D/W2", "P(score)", "Prediction"],
        tablefmt="rounded_outline",
    ))

    # 7. OUTPUTS
    print("\n[6/6] Saving outputs...")
    print("\n  === MODEL COMPARISON (LOO-CV) ===")
    metrics_rows = []
    for name in builders:
        b, h = cv_base[name], cv_hist[name]
        metrics_rows.append([
            name,
            b["mae"], h["mae"], f"{b['mae'] - h['mae']:+.4f}",
            f"{b['result_acc_pct']}%", f"{h['result_acc_pct']}%",
            f"{h['result_acc_pct'] - b['result_acc_pct']:+.1f}%",
            f"{h['exact_score_pct']}%",
        ])
    print(tabulate(
        metrics_rows,
        headers=["Model", "MAE base", "MAE hist", "ΔMAE",
                 "Acc% base", "Acc% hist", "ΔAcc%", "Exact score"],
        tablefmt="rounded_outline", floatfmt=".4f",
    ))

    # CSV
    df_out = pd.DataFrame([{
        "Date":                p["date"],
        "Group":               p["group"],
        "Team1":               p["team1"],
        "Predicted_Goals_T1":  p["g1"],
        "Predicted_Goals_T2":  p["g2"],
        "Team2":               p["team2"],
        "Lambda_T1":           p["lambda1"],
        "Lambda_T2":           p["lambda2"],
        "P_Win_T1":            round(p["p_win1"]     * 100, 1),
        "P_Draw":              round(p["p_draw"]     * 100, 1),
        "P_Win_T2":            round(p["p_win2"]     * 100, 1),
        "P_Exact_Score":       round(p["score_prob"] * 100, 2),
        "Predicted_Result":    _result_label(p["g1"], p["g2"], p["team1"], p["team2"]),
    } for p in predictions])
    csv_path = OUTPUT_DIR / "predictions.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\n  CSV: {csv_path}")

    # Charts
    fi_path = CHARTS_DIR / "feature_importance.png"
    plot_feature_importance(xgb_final, fi_path)
    print(f"  Chart: {fi_path}")

    hm_path = CHARTS_DIR / "score_heatmaps.png"
    plot_score_heatmaps(predictions, rho, hm_path)
    print(f"  Chart: {hm_path}")

    rp_path = CHARTS_DIR / "result_probabilities.png"
    plot_result_probs(predictions, rp_path)
    print(f"  Chart: {rp_path}")

    cmp_path = CHARTS_DIR / "model_comparison.png"
    plot_model_comparison(cv_base, cv_hist, cmp_path)
    print(f"  Chart: {cmp_path}")

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
