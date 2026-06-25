"""
Avaliação dos modelos via Leave-One-Out Cross-Validation (LOO-CV).

Com apenas 48 partidas, LOO-CV é preferível ao k-fold.
Métricas: MAE, RMSE, acurácia de resultado (V1/E/V2), gol exato, placar exato.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features import (
    FEATURE_COLS,
    _row_features,
    build_training_data,
    compute_team_stats,
)


def _result(g1: int, g2: int) -> str:
    if g1 > g2:
        return "V1"
    if g1 < g2:
        return "V2"
    return "E"


def loo_cv(
    played: pd.DataFrame,
    rankings: dict,
    model_builder,
    historical: Optional[pd.DataFrame] = None,
    ref_date: Optional[pd.Timestamp] = None,
) -> dict:
    """
    Executa LOO-CV para um callable `model_builder()` que retorna
    um estimador sklearn (com .fit e .predict).

    historical: DataFrame de histórico pré-Copa (opcional).
    ref_date: data de referência para o decay (default: 2026-06-24).
    """
    true_g1, true_g2, pred_g1, pred_g2 = [], [], [], []

    for idx in played.index:
        X_train, y_train = build_training_data(
            played, rankings,
            loo_idx=idx,
            historical=historical,
            ref_date=ref_date,
        )
        model = model_builder()
        model.fit(X_train, y_train)

        # Features para a partida excluída
        row = played.loc[idx]
        ts = compute_team_stats(
            played,
            exclude_idx=idx,
            historical=historical,
            ref_date=ref_date,
        )
        f1 = _row_features(row["time1"], row["time2"], ts, rankings)
        f2 = _row_features(row["time2"], row["time1"], ts, rankings)
        X_pred = pd.DataFrame([f1, f2], columns=FEATURE_COLS)

        preds = np.clip(model.predict(X_pred), 0, None)
        pred_g1.append(preds[0])
        pred_g2.append(preds[1])
        true_g1.append(row["gols1"])
        true_g2.append(row["gols2"])

    true_all = np.array(true_g1 + true_g2, dtype=float)
    pred_all = np.array(pred_g1 + pred_g2, dtype=float)

    mae  = mean_absolute_error(true_all, pred_all)
    rmse = float(np.sqrt(mean_squared_error(true_all, pred_all)))

    exact_goal = np.mean(np.round(pred_all) == true_all)

    exact_score = np.mean([
        (round(p1) == t1 and round(p2) == t2)
        for p1, t1, p2, t2 in zip(pred_g1, true_g1, pred_g2, true_g2)
    ])

    result_acc = np.mean([
        _result(round(p1), round(p2)) == _result(t1, t2)
        for p1, t1, p2, t2 in zip(pred_g1, true_g1, pred_g2, true_g2)
    ])

    return {
        "mae":             round(mae, 4),
        "rmse":            round(rmse, 4),
        "exact_goal_pct":  round(float(exact_goal)  * 100, 1),
        "exact_score_pct": round(float(exact_score) * 100, 1),
        "result_acc_pct":  round(float(result_acc)  * 100, 1),
    }
