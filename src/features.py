"""
Engenharia de features para previsão de placares.

Estratégia:
  - Cada partida gera 2 linhas de treino (uma por perspectiva de ataque).
  - Com histórico disponível, as taxas de ataque/defesa são calculadas como
    média ponderada de dois sinais:
      (a) Histórico pré-Copa (2020-2026): decay exponencial por tempo
          + peso do torneio (Copa > Eliminatórias > Conf. > Amistoso)
      (b) Jogos da fase de grupos atual: peso fixo equivalente a Copa do Mundo
  - Sem histórico, usa suavização bayesiana simples com prior = média da liga.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "attack_off",     # taxa de ataque do time que marca (gols/jogo ponderado)
    "defense_def",    # taxa defensiva do adversário (gols sofridos/jogo ponderado)
    "rank_off",       # ranking FIFA do atacante (÷1000)
    "rank_def",       # ranking FIFA do defensor (÷1000)
    "pts_off",        # pontos/jogo do atacante (fase de grupos)
    "pts_def",        # pontos/jogo do defensor (fase de grupos)
    "rank_diff",      # diferença de ranking (off - def) ÷1000
    "atk_vs_def",     # attack_off - defense_def
    "quality_ratio",  # rank_off / rank_def
]

_AVG_GOALS   = 1.48   # média global de gols por time por jogo
_PRIOR_K     = 2      # jogos fictícios de prior bayesiano (fallback sem histórico)
_DECAY_RATE  = 0.003  # λ do decay temporal: peso = exp(-λ * dias)
_COPA_W      = 3.0    # peso-torneio para jogos da Copa do Mundo atual
_REF_DATE    = pd.Timestamp("2026-06-24")  # data de referência para previsões


def _smoothed_rate(raw: float, games: int, prior: float = _AVG_GOALS, k: int = _PRIOR_K) -> float:
    """Suavização bayesiana usada quando não há histórico disponível."""
    return (raw * games + prior * k) / (games + k)


# ─────────────────────────────────────────────────────────────────────────────
# Cálculo de estatísticas por seleção
# ─────────────────────────────────────────────────────────────────────────────

def compute_team_stats(
    matches: pd.DataFrame,
    exclude_idx: Optional[int] = None,
    historical: Optional[pd.DataFrame] = None,
    ref_date: Optional[pd.Timestamp] = None,
    decay_rate: float = _DECAY_RATE,
) -> dict:
    """
    Calcula estatísticas de ataque/defesa por seleção.

    Sem histórico: suavização bayesiana sobre jogos da fase de grupos.
    Com histórico: média ponderada de (histórico com decay) + (Copa atual).

    exclude_idx: exclui essa linha de matches para LOO-CV sem data leakage.
    ref_date: data de referência para o decay temporal (default: hoje).
    """
    if ref_date is None:
        ref_date = _REF_DATE

    # ── Parte 1: histórico pré-Copa (se disponível) ──────────────────────
    hist_w: dict = defaultdict(lambda: {"wgf": 0.0, "wgc": 0.0, "wtot": 0.0})

    if historical is not None and not historical.empty:
        for _, row in historical.iterrows():
            days_ago = (ref_date - row["data"]).days
            if days_ago < 0:
                continue
            w = np.exp(-decay_rate * days_ago) * float(row["torneio_weight"])
            hist_w[row["team"]]["wgf"]  += row["gols_for"]    * w
            hist_w[row["team"]]["wgc"]  += row["gols_against"] * w
            hist_w[row["team"]]["wtot"] += w

    # ── Parte 2: jogos da fase de grupos atual ───────────────────────────
    copa: dict = defaultdict(lambda: {"gp": 0, "gf": 0, "gc": 0, "pts": 0})

    for idx, row in matches.iterrows():
        if exclude_idx is not None and idx == exclude_idx:
            continue
        for team, gf, gc in [
            (row["time1"], int(row["gols1"]), int(row["gols2"])),
            (row["time2"], int(row["gols2"]), int(row["gols1"])),
        ]:
            copa[team]["gp"] += 1
            copa[team]["gf"] += gf
            copa[team]["gc"] += gc
            if gf > gc:
                copa[team]["pts"] += 3
            elif gf == gc:
                copa[team]["pts"] += 1

    # ── Parte 3: combinar ────────────────────────────────────────────────
    all_teams = set(hist_w) | set(copa)
    result: dict = {}

    for team in all_teams:
        hs = hist_w[team]
        cs = copa[team]
        gp = cs["gp"]

        if hs["wtot"] > 0:
            # Histórico disponível: blend ponderado
            h_attack  = hs["wgf"] / hs["wtot"]
            h_defense = hs["wgc"] / hs["wtot"]

            # Copa atual: cada jogo vale _COPA_W unidades de peso
            c_weight = gp * _COPA_W
            if gp > 0:
                c_attack  = cs["gf"] / gp
                c_defense = cs["gc"] / gp
            else:
                c_attack  = h_attack
                c_defense = h_defense
                c_weight  = 0.0

            total_w = hs["wtot"] + c_weight
            attack_rate  = (h_attack  * hs["wtot"] + c_attack  * c_weight) / total_w
            defense_rate = (h_defense * hs["wtot"] + c_defense * c_weight) / total_w
        else:
            # Fallback: suavização bayesiana simples
            raw_atk = cs["gf"] / gp if gp else _AVG_GOALS
            raw_def = cs["gc"] / gp if gp else _AVG_GOALS
            attack_rate  = _smoothed_rate(raw_atk, gp)
            defense_rate = _smoothed_rate(raw_def, gp)

        result[team] = {
            "attack_rate":  attack_rate,
            "defense_rate": defense_rate,
            "pts_per_game": cs["pts"] / gp if gp else 1.0,
            "gp":           gp,
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Construção do vetor de features por perspectiva de ataque
# ─────────────────────────────────────────────────────────────────────────────

def _row_features(
    team_off: str,
    team_def: str,
    team_stats: dict,
    rankings: dict,
) -> dict:
    default = {"attack_rate": _AVG_GOALS, "defense_rate": _AVG_GOALS, "pts_per_game": 1.0}
    s_off = team_stats.get(team_off, default)
    s_def = team_stats.get(team_def, default)

    avg_rank = 1550.0
    r_off = rankings.get(team_off, avg_rank) / 1000.0
    r_def = rankings.get(team_def, avg_rank) / 1000.0

    return {
        "attack_off":   s_off["attack_rate"],
        "defense_def":  s_def["defense_rate"],
        "rank_off":     r_off,
        "rank_def":     r_def,
        "pts_off":      s_off["pts_per_game"],
        "pts_def":      s_def["pts_per_game"],
        "rank_diff":    r_off - r_def,
        "atk_vs_def":  s_off["attack_rate"] - s_def["defense_rate"],
        "quality_ratio": r_off / max(r_def, 0.001),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Montagem dos datasets de treino e previsão
# ─────────────────────────────────────────────────────────────────────────────

def build_training_data(
    played: pd.DataFrame,
    rankings: dict,
    loo_idx: Optional[int] = None,
    historical: Optional[pd.DataFrame] = None,
    ref_date: Optional[pd.Timestamp] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Constrói X (features) e y (gols) para treino.
    Cada partida gera 2 linhas — uma por time.
    loo_idx exclui a partida das stats E do treino (LOO-CV limpo).
    """
    team_stats = compute_team_stats(
        played, exclude_idx=loo_idx, historical=historical, ref_date=ref_date
    )
    X_rows, y_rows = [], []

    for idx, row in played.iterrows():
        if loo_idx is not None and idx == loo_idx:
            continue
        t1, t2 = row["time1"], row["time2"]
        X_rows.append(_row_features(t1, t2, team_stats, rankings))
        y_rows.append(float(row["gols1"]))
        X_rows.append(_row_features(t2, t1, team_stats, rankings))
        y_rows.append(float(row["gols2"]))

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
    Retorna (X_time1, X_time2) para as partidas futuras,
    usando estatísticas de todos os jogos disputados + histórico completo.
    """
    team_stats = compute_team_stats(
        played, historical=historical, ref_date=ref_date
    )
    rows1, rows2 = [], []

    for _, row in future.iterrows():
        t1, t2 = row["time1"], row["time2"]
        rows1.append(_row_features(t1, t2, team_stats, rankings))
        rows2.append(_row_features(t2, t1, team_stats, rankings))

    X1 = pd.DataFrame(rows1, columns=FEATURE_COLS)
    X2 = pd.DataFrame(rows2, columns=FEATURE_COLS)
    return X1, X2
