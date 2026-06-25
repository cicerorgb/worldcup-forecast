"""
Copa do Mundo 2026 — Previsão de Placares (3ª Rodada da Fase de Grupos)

Pipeline:
  1. Carrega dados (rankings FIFA + partidas + histórico HTML)
  2. Engenharia de features com decay temporal sobre histórico 2020-2026
  3. LOO-CV em 3 modelos com e sem histórico (comparativo de melhoria)
  4. Ensemble ponderado pelo inverso do MAE
  5. Estima ρ de Dixon-Coles
  6. Gera previsões + probabilidades (V1/E/V2) + placar mais provável
  7. Salva CSV e gráficos em outputs/
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

OUTPUT_DIR   = Path("outputs")
GRAFICOS_DIR = OUTPUT_DIR / "graficos"
OUTPUT_DIR.mkdir(exist_ok=True)
GRAFICOS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────────────────────────────────────

def _result_label(g1: int, g2: int, t1: str, t2: str) -> str:
    if g1 > g2:
        return f"Vitória {t1}"
    if g1 < g2:
        return f"Vitória {t2}"
    return "Empate"


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(xgb_model, path: Path) -> None:
    try:
        reg = xgb_model.named_steps["reg"] if hasattr(xgb_model, "named_steps") else xgb_model
        importances = reg.feature_importances_
    except AttributeError:
        importances = xgb_model.feature_importances_

    fi = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True)
    labels_pt = {
        "attack_off":   "Taxa de Ataque (atacante)",
        "defense_def":  "Taxa Defensiva (adversário)",
        "rank_off":     "Ranking FIFA (atacante)",
        "rank_def":     "Ranking FIFA (adversário)",
        "pts_off":      "Pontos/Jogo (atacante)",
        "pts_def":      "Pontos/Jogo (adversário)",
        "rank_diff":    "Diferença de Ranking",
        "atk_vs_def":  "Ataque vs Defesa",
        "quality_ratio":"Razão de Qualidade",
    }
    fi.index = [labels_pt.get(c, c) for c in fi.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1a73e8" if v >= fi.median() else "#90caf9" for v in fi.values]
    fi.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Importância (gain)", fontsize=11)
    ax.set_title("Importância das Features — XGBoost (com histórico)", fontsize=13, fontweight="bold")
    ax.axvline(fi.median(), color="red", linestyle="--", alpha=0.6, label="Mediana")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_score_heatmaps(predictions: list[dict], rho: float, path: Path) -> None:
    highlights = ["Brasil", "Argentina", "Espanha", "França", "Inglaterra", "Portugal"]
    selected = [p for p in predictions if p["time1"] in highlights or p["time2"] in highlights][:6]
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
        ax.set_xlabel(pred["time2"], fontsize=10, fontweight="bold")
        ax.set_ylabel(pred["time1"], fontsize=10, fontweight="bold")
        ax.set_title(
            f"Grupo {pred['grupo']} | {pred['time1']} vs {pred['time2']}\n"
            f"Placar mais provável: {pred['g1']}x{pred['g2']}",
            fontsize=9, fontweight="bold",
        )

    for j in range(len(selected), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Probabilidades de Placar (%) — 3ª Rodada",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_result_probs(predictions: list[dict], path: Path) -> None:
    n = len(predictions)
    fig, ax = plt.subplots(figsize=(14, n * 0.55 + 1.5))

    labels = [f"G{p['grupo']}: {p['time1']} vs {p['time2']}" for p in predictions]
    pw  = [p["p_v1"]  for p in predictions]
    pd_ = [p["p_emp"] for p in predictions]
    pl  = [p["p_v2"]  for p in predictions]
    y   = np.arange(n)
    h   = 0.6

    ax.barh(y, pw,       height=h, color="#1a73e8", label="Vitória Time 1")
    ax.barh(y, pd_, left=pw, height=h, color="#e0e0e0", label="Empate")
    ax.barh(y, pl,  left=[a + b for a, b in zip(pw, pd_)], height=h, color="#e53935", label="Vitória Time 2")

    for i, (p1, pe, p2) in enumerate(zip(pw, pd_, pl)):
        if p1 > 0.12:
            ax.text(p1 / 2, i, f"{p1:.0%}", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
        if pe > 0.08:
            ax.text(p1 + pe / 2, i, f"{pe:.0%}", ha="center", va="center", fontsize=7.5, color="#444")
        if p2 > 0.12:
            ax.text(p1 + pe + p2 / 2, i, f"{p2:.0%}", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilidade", fontsize=10)
    ax.set_title("Probabilidades de Resultado — 3ª Rodada (com histórico)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_improvement(baseline: dict[str, dict], with_hist: dict[str, dict], path: Path) -> None:
    """Gráfico de barras comparando MAE e acurácia de resultado com/sem histórico."""
    models = list(baseline.keys())
    x = np.arange(len(models))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    mae_base = [baseline[m]["mae"] for m in models]
    mae_hist = [with_hist[m]["mae"] for m in models]
    acc_base = [baseline[m]["result_acc_pct"] for m in models]
    acc_hist = [with_hist[m]["result_acc_pct"] for m in models]

    ax1.bar(x - w/2, mae_base, w, label="Sem histórico", color="#90caf9")
    ax1.bar(x + w/2, mae_hist, w, label="Com histórico", color="#1a73e8")
    ax1.set_xticks(x); ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylabel("MAE (gols)", fontsize=10)
    ax1.set_title("Erro Médio Absoluto (menor = melhor)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    for xi, (b, h) in enumerate(zip(mae_base, mae_hist)):
        delta = b - h
        color = "green" if delta > 0 else "red"
        ax1.text(xi + w/2, h + 0.01, f"{delta:+.3f}", ha="center", va="bottom", fontsize=8, color=color)

    ax2.bar(x - w/2, acc_base, w, label="Sem histórico", color="#ef9a9a")
    ax2.bar(x + w/2, acc_hist, w, label="Com histórico", color="#e53935")
    ax2.set_xticks(x); ax2.set_xticklabels(models, fontsize=9)
    ax2.set_ylabel("Acurácia de Resultado (%)", fontsize=10)
    ax2.set_title("Resultado Certo V1/E/V2 (maior = melhor)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    for xi, (b, h) in enumerate(zip(acc_base, acc_hist)):
        delta = h - b
        color = "green" if delta > 0 else "red"
        ax2.text(xi + w/2, h + 0.3, f"{delta:+.1f}%", ha="center", va="bottom", fontsize=8, color=color)

    fig.suptitle("Impacto do Histórico 2020-2026 nos Modelos (LOO-CV)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  COPA DO MUNDO 2026 — PREVISÃO DE PLACARES (3ª RODADA)")
    print("=" * 65)

    # 1. DADOS
    print("\n[1/6] Carregando dados...")
    rankings = load_rankings()
    played, future = load_matches()

    historical_raw = load_historical(year_from=2020)
    # Excluir jogos da Copa 2026 do histórico (tratados separadamente via played)
    historical = historical_raw[
        ~historical_raw["torneio"].str.contains("Copa do Mundo da FIFA 2026", na=False)
    ].copy()

    jogos_por_time = historical.groupby("team").size()
    print(f"      {len(played)} partidas na Copa | {len(future)} a prever")
    print(f"      {len(rankings)} seleções com ranking FIFA")
    print(f"      {len(historical)} registros históricos ({historical['data'].dt.year.min()}-2026)")
    print(f"      Media de {jogos_por_time.mean():.0f} jogos por seleção no histórico")

    builders = {
        "Poisson GLM": build_poisson,
        "XGBoost":     build_xgb,
        "LightGBM":    build_lgb,
    }

    # 2. LOO-CV SEM HISTÓRICO (baseline)
    print("\n[2/6] LOO-CV sem histórico (baseline)...")
    cv_base: dict[str, dict] = {}
    for name, builder in builders.items():
        print(f"      -> {name}...", end=" ", flush=True)
        cv = loo_cv(played, rankings, builder, historical=None)
        cv_base[name] = cv
        print(f"MAE={cv['mae']:.3f} | Resultado={cv['result_acc_pct']:.1f}% | Placar={cv['exact_score_pct']:.1f}%")

    # 3. LOO-CV COM HISTÓRICO
    print("\n[3/6] LOO-CV com histórico (2020-2026)...")
    cv_hist: dict[str, dict] = {}
    for name, builder in builders.items():
        print(f"      -> {name}...", end=" ", flush=True)
        cv = loo_cv(played, rankings, builder, historical=historical)
        cv_hist[name] = cv
        print(f"MAE={cv['mae']:.3f} | Resultado={cv['result_acc_pct']:.1f}% | Placar={cv['exact_score_pct']:.1f}%")

    # Comparativo
    print("\n      Ganho com histórico:")
    for name in builders:
        delta_mae = cv_base[name]["mae"] - cv_hist[name]["mae"]
        delta_acc = cv_hist[name]["result_acc_pct"] - cv_base[name]["result_acc_pct"]
        sign_mae = "melhora" if delta_mae > 0 else "piora"
        sign_acc = "melhora" if delta_acc > 0 else "piora"
        print(f"        {name}: MAE {delta_mae:+.3f} ({sign_mae}) | "
              f"Resultado {delta_acc:+.1f}% ({sign_acc})")

    # 4. PESOS ENSEMBLE (inverso MAE com histórico)
    maes = {n: cv_hist[n]["mae"] for n in builders}
    inv_sum = sum(1 / v for v in maes.values())
    weights = [1 / maes[n] / inv_sum for n in builders]
    print(f"\n      Pesos ensemble: " +
          " | ".join(f"{n}={w:.2f}" for n, w in zip(builders.keys(), weights)))

    # 5. MODELOS FINAIS
    print("\n[4/6] Treinando modelos finais em 48 partidas + histórico...")
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
    print(f"      rho Dixon-Coles estimado: {rho:.4f}")

    # 6. PREVISÕES
    print("\n[5/6] Gerando previsões para a 3ª rodada...")
    X_fut1, X_fut2 = build_prediction_features(future, played, rankings, historical=historical)
    lambda1_all = np.clip(ensemble.predict(X_fut1), 0.05, None)
    lambda2_all = np.clip(ensemble.predict(X_fut2), 0.05, None)

    predictions = []
    for i, (_, row) in enumerate(future.iterrows()):
        l1, l2 = float(lambda1_all[i]), float(lambda2_all[i])
        g1, g2, prob = most_likely_score(l1, l2, rho)
        pw, pe, pv2  = result_probabilities(l1, l2, rho)
        predictions.append({
            "data":    row["data"].strftime("%d/%m"),
            "grupo":   row["grupo"],
            "time1":   row["time1"],
            "time2":   row["time2"],
            "g1": g1,  "g2": g2,
            "lambda1": round(l1, 3),
            "lambda2": round(l2, 3),
            "p_v1":    pw, "p_emp": pe, "p_v2": pv2,
            "prob_placar": prob,
        })

    print()
    table_rows = [[
        p["data"], p["grupo"], p["time1"],
        f"{p['g1']} x {p['g2']}",
        p["time2"],
        f"λ {p['lambda1']:.2f}x{p['lambda2']:.2f}",
        f"{p['p_v1']:.0%} / {p['p_emp']:.0%} / {p['p_v2']:.0%}",
        f"{p['prob_placar']:.1%}",
        _result_label(p["g1"], p["g2"], p["time1"], p["time2"]),
    ] for p in predictions]

    print(tabulate(
        table_rows,
        headers=["Data", "Grp", "Time 1", "Placar", "Time 2",
                 "Exp. Gols", "V1/E/V2", "P(placar)", "Resultado"],
        tablefmt="rounded_outline",
    ))

    # Métricas finais (com histórico)
    print("\n[6/6] Salvando outputs...")
    print("\n  === COMPARATIVO FINAL DE MODELOS (LOO-CV) ===")
    metrics_rows = []
    for name in builders:
        b, h = cv_base[name], cv_hist[name]
        metrics_rows.append([
            name,
            b["mae"], h["mae"], f"{b['mae']-h['mae']:+.4f}",
            f"{b['result_acc_pct']}%", f"{h['result_acc_pct']}%",
            f"{h['result_acc_pct']-b['result_acc_pct']:+.1f}%",
            f"{h['exact_score_pct']}%",
        ])
    print(tabulate(
        metrics_rows,
        headers=["Modelo", "MAE base", "MAE hist", "Delta MAE",
                 "Res% base", "Res% hist", "Delta Res%", "Placar exato"],
        tablefmt="rounded_outline", floatfmt=".4f",
    ))

    # CSV
    df_out = pd.DataFrame([{
        "Data":           p["data"],
        "Grupo":          p["grupo"],
        "Time1":          p["time1"],
        "Gols_Prev_T1":   p["g1"],
        "Gols_Prev_T2":   p["g2"],
        "Time2":          p["time2"],
        "Lambda_T1":      p["lambda1"],
        "Lambda_T2":      p["lambda2"],
        "P_Vitoria_T1":   round(p["p_v1"]  * 100, 1),
        "P_Empate":       round(p["p_emp"] * 100, 1),
        "P_Vitoria_T2":   round(p["p_v2"]  * 100, 1),
        "P_Placar_Exato": round(p["prob_placar"] * 100, 2),
        "Resultado_Prev": _result_label(p["g1"], p["g2"], p["time1"], p["time2"]),
    } for p in predictions])
    csv_path = OUTPUT_DIR / "predicoes.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\n  CSV: {csv_path}")

    # Gráficos
    fi_path = GRAFICOS_DIR / "feature_importance.png"
    plot_feature_importance(xgb_final, fi_path)
    print(f"  Grafico: {fi_path}")

    hm_path = GRAFICOS_DIR / "score_heatmaps.png"
    plot_score_heatmaps(predictions, rho, hm_path)
    print(f"  Grafico: {hm_path}")

    rp_path = GRAFICOS_DIR / "result_probabilities.png"
    plot_result_probs(predictions, rp_path)
    print(f"  Grafico: {rp_path}")

    imp_path = GRAFICOS_DIR / "improvement_historico.png"
    plot_improvement(cv_base, cv_hist, imp_path)
    print(f"  Grafico: {imp_path}")

    print("\n" + "=" * 65)
    print("  CONCLUIDO")
    print("=" * 65)


if __name__ == "__main__":
    main()
