"""
Feature engineering for match score prediction.

Temporal weighting strategy (inspired by FIFA ranking pre-2018 methodology):
  Results are grouped into 6-month windows. Each window carries a progressively
  lower weight, giving recent form much more influence than distant results.
  This is analogous to the stepped multipliers used in football ranking systems
  before continuous Elo was adopted (Hvattum & Arntzen, 2010).

  Within each bucket the decay is already captured by the bucket weight itself;
  no additional intra-bucket smoothing is applied, keeping the scheme interpretable.

Asymmetric loss penalty:
  When a team *lost* a historical match, the goals conceded in that game receive
  an extra recency multiplier before being averaged into the defence rate.
  This reflects a well-known football-forecasting heuristic: a recent heavy defeat
  is a stronger signal of current defensive fragility than an old one, because
  squads change, tactics evolve, and morale fluctuates. Older losses carry
  progressively less extra weight until they are treated symmetrically with draws
  and wins (multiplier = 1.0 beyond 18 months).

Blend:
  When historical data is available, attack/defence rates are a weighted blend of:
    (a) Pre-tournament history (2020–2026): 6-month bucket weight × tournament weight
        with asymmetric loss penalty on defence rate
    (b) In-tournament group-stage matches: fixed weight = _COPA_W (World Cup level)
  Without historical data, Bayesian smoothing towards a league-average prior is used.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "attack_off",     # weighted goal-scoring rate of the attacking team
    "defense_def",    # weighted goals-conceded rate of the defending team
    "rank_off",       # FIFA points of the attacking team (÷1000)
    "rank_def",       # FIFA points of the defending team (÷1000)
    "pts_off",        # points per game of the attacking team (group stage)
    "pts_def",        # points per game of the defending team (group stage)
    "rank_diff",      # (rank_off - rank_def) ÷ 1000
    "atk_vs_def",     # attack_off - defense_def
    "quality_ratio",  # rank_off / rank_def
]

_AVG_GOALS = 1.48   # global baseline goals per team per match
_PRIOR_K   = 2      # pseudo-matches used in Bayesian prior (no-history fallback)
_COPA_W    = 3.0    # tournament weight for current World Cup matches
_REF_DATE  = pd.Timestamp("2026-06-24")  # reference date for bucket computation

# 6-month stepped weights.
# Each tuple: (lower_bound_months, upper_bound_months, weight_multiplier)
# Calibrated so the midpoint of each bucket matches an exponential curve
# with a ~12-month half-life (ξ ≈ 0.0019/day, see Dixon-Coles practitioners).
_TEMPORAL_BUCKETS: list[tuple[float, float, float]] = [
    (0,   6,   1.00),   # 0–6 months:   full weight
    (6,   12,  0.70),   # 6–12 months:  70 %
    (12,  18,  0.50),   # 12–18 months: 50 %
    (18,  24,  0.35),   # 18–24 months: 35 %
    (24,  30,  0.25),   # 24–30 months: 25 %
    (30,  float("inf"), 0.15),  # 30+ months:   15 %
]

# Extra multiplier applied to goals_against when the team LOST that match.
# Only applied within the recency window where losses carry the strongest signal.
# Each tuple: (lower_bound_months, upper_bound_months, defence_penalty)
_LOSS_RECENCY_MULTIPLIERS: list[tuple[float, float, float]] = [
    (0,   6,   1.50),   # 0–6 months:   losses count 50 % extra on defence rate
    (6,   12,  1.25),   # 6–12 months:  25 % extra
    (12,  18,  1.10),   # 12–18 months: 10 % extra
    (18,  float("inf"), 1.00),  # older losses: no additional penalty
]


def _months_ago(days: int) -> float:
    return days / 30.4375  # average days per month (365.25 / 12)


def _temporal_weight(months: float) -> float:
    """Returns the 6-month bucket weight for a match that occurred `months` ago."""
    for lo, hi, w in _TEMPORAL_BUCKETS:
        if lo <= months < hi:
            return w
    return _TEMPORAL_BUCKETS[-1][2]


def _loss_defence_multiplier(months: float, is_loss: bool) -> float:
    """
    Returns the extra multiplier applied to goals_against for a lost match.
    For non-losses (wins or draws) always returns 1.0.
    """
    if not is_loss:
        return 1.0
    for lo, hi, m in _LOSS_RECENCY_MULTIPLIERS:
        if lo <= months < hi:
            return m
    return 1.0


def _smoothed_rate(raw: float, games: int, prior: float = _AVG_GOALS, k: int = _PRIOR_K) -> float:
    """Bayesian smoothing towards the league-average prior (used when no history is available)."""
    return (raw * games + prior * k) / (games + k)


# ─────────────────────────────────────────────────────────────────────────────
# Per-team statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_team_stats(
    matches: pd.DataFrame,
    exclude_idx: Optional[int] = None,
    historical: Optional[pd.DataFrame] = None,
    ref_date: Optional[pd.Timestamp] = None,
) -> dict:
    """
    Computes attack and defence rates for each team.

    Without history: Bayesian smoothing over group-stage matches only.
    With history: weighted blend of (6-month-bucketed historical data) +
                  (group-stage data at World Cup weight).

    The defence rate applies an asymmetric recency penalty to goals conceded
    in losses: recent defeats inflate the defence rate more than old ones.

    exclude_idx: omits this match index from both stats and training (clean LOO-CV).
    ref_date: reference date for bucket assignment (defaults to _REF_DATE).
    """
    if ref_date is None:
        ref_date = _REF_DATE

    # ── Part 1: pre-tournament historical data (if available) ────────────────
    hist_w: dict = defaultdict(lambda: {"wgf": 0.0, "wgc": 0.0, "wtot": 0.0})

    if historical is not None and not historical.empty:
        for _, row in historical.iterrows():
            days_ago = (ref_date - row["date"]).days
            if days_ago < 0:
                continue

            months = _months_ago(days_ago)
            bucket_w = _temporal_weight(months)
            w = bucket_w * float(row["tournament_weight"])

            is_loss = row["goals_for"] < row["goals_against"]
            loss_mult = _loss_defence_multiplier(months, is_loss)

            hist_w[row["team"]]["wgf"]  += row["goals_for"]    * w
            hist_w[row["team"]]["wgc"]  += row["goals_against"] * w * loss_mult
            hist_w[row["team"]]["wtot"] += w

    # ── Part 2: current group-stage matches ──────────────────────────────────
    copa: dict = defaultdict(lambda: {"gp": 0, "gf": 0, "gc": 0, "pts": 0})

    for idx, row in matches.iterrows():
        if exclude_idx is not None and idx == exclude_idx:
            continue
        for team, gf, gc in [
            (row["team1"], int(row["goals1"]), int(row["goals2"])),
            (row["team2"], int(row["goals2"]), int(row["goals1"])),
        ]:
            copa[team]["gp"] += 1
            copa[team]["gf"] += gf
            copa[team]["gc"] += gc
            if gf > gc:
                copa[team]["pts"] += 3
            elif gf == gc:
                copa[team]["pts"] += 1

    # ── Part 3: blend ────────────────────────────────────────────────────────
    all_teams = set(hist_w) | set(copa)
    result: dict = {}

    for team in all_teams:
        hs = hist_w[team]
        cs = copa[team]
        gp = cs["gp"]

        if hs["wtot"] > 0:
            h_attack  = hs["wgf"] / hs["wtot"]
            h_defence = hs["wgc"] / hs["wtot"]

            # Each group-stage match counts as _COPA_W weight units
            c_weight = gp * _COPA_W
            if gp > 0:
                c_attack  = cs["gf"] / gp
                c_defence = cs["gc"] / gp
            else:
                c_attack  = h_attack
                c_defence = h_defence
                c_weight  = 0.0

            total_w      = hs["wtot"] + c_weight
            attack_rate  = (h_attack  * hs["wtot"] + c_attack  * c_weight) / total_w
            defence_rate = (h_defence * hs["wtot"] + c_defence * c_weight) / total_w
        else:
            raw_atk = cs["gf"] / gp if gp else _AVG_GOALS
            raw_def = cs["gc"] / gp if gp else _AVG_GOALS
            attack_rate  = _smoothed_rate(raw_atk, gp)
            defence_rate = _smoothed_rate(raw_def, gp)

        result[team] = {
            "attack_rate":   attack_rate,
            "defence_rate":  defence_rate,
            "pts_per_game":  cs["pts"] / gp if gp else 1.0,
            "gp":            gp,
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector construction
# ─────────────────────────────────────────────────────────────────────────────

def _row_features(
    team_off: str,
    team_def: str,
    team_stats: dict,
    rankings: dict,
) -> dict:
    default = {"attack_rate": _AVG_GOALS, "defence_rate": _AVG_GOALS, "pts_per_game": 1.0}
    s_off = team_stats.get(team_off, default)
    s_def = team_stats.get(team_def, default)

    avg_rank = 1550.0
    r_off = rankings.get(team_off, avg_rank) / 1000.0
    r_def = rankings.get(team_def, avg_rank) / 1000.0

    return {
        "attack_off":    s_off["attack_rate"],
        "defense_def":   s_def["defence_rate"],
        "rank_off":      r_off,
        "rank_def":      r_def,
        "pts_off":       s_off["pts_per_game"],
        "pts_def":       s_def["pts_per_game"],
        "rank_diff":     r_off - r_def,
        "atk_vs_def":    s_off["attack_rate"] - s_def["defence_rate"],
        "quality_ratio": r_off / max(r_def, 0.001),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builders
# ─────────────────────────────────────────────────────────────────────────────

def build_training_data(
    played: pd.DataFrame,
    rankings: dict,
    loo_idx: Optional[int] = None,
    historical: Optional[pd.DataFrame] = None,
    ref_date: Optional[pd.Timestamp] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Builds X (feature matrix) and y (goal counts) for training.
    Each match generates 2 rows — one per team perspective.
    loo_idx excludes that match from both team stats and the training set (clean LOO-CV).
    """
    team_stats = compute_team_stats(
        played, exclude_idx=loo_idx, historical=historical, ref_date=ref_date
    )
    X_rows, y_rows = [], []

    for idx, row in played.iterrows():
        if loo_idx is not None and idx == loo_idx:
            continue
        t1, t2 = row["team1"], row["team2"]
        X_rows.append(_row_features(t1, t2, team_stats, rankings))
        y_rows.append(float(row["goals1"]))
        X_rows.append(_row_features(t2, t1, team_stats, rankings))
        y_rows.append(float(row["goals2"]))

    X = pd.DataFrame(X_rows, columns=FEATURE_COLS)
    y = np.array(y_rows, dtype=float)
    return X, y


def build_prediction_features(
    future: pd.DataFrame,
    played: pd.DataFrame,
    rankings: dict,
    historical: Optional[pd.DataFrame] = None,
    ref_date: Optional[pd.Timestamp] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (X_team1, X_team2) for unplayed matches, using all played matches
    and the full historical dataset to compute team statistics.
    """
    team_stats = compute_team_stats(
        played, historical=historical, ref_date=ref_date
    )
    rows1, rows2 = [], []

    for _, row in future.iterrows():
        t1, t2 = row["team1"], row["team2"]
        rows1.append(_row_features(t1, t2, team_stats, rankings))
        rows2.append(_row_features(t2, t1, team_stats, rankings))

    X1 = pd.DataFrame(rows1, columns=FEATURE_COLS)
    X2 = pd.DataFrame(rows2, columns=FEATURE_COLS)
    return X1, X2
