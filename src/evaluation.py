"""
Leave-One-Out Cross-Validation (LOO-CV) harness.

With only 48 matches, LOO-CV is preferred over k-fold to maximise the training
set size at each fold. Metrics reported: MAE, RMSE, result accuracy (W/D/L),
exact goal count accuracy, and exact score accuracy.
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
        return "W1"
    if g1 < g2:
        return "W2"
    return "D"


def loo_cv(
    played: pd.DataFrame,
    rankings: dict,
    model_builder,
    historical: Optional[pd.DataFrame] = None,
    ref_date: Optional[pd.Timestamp] = None,
) -> dict:
    """
    Runs LOO-CV for a `model_builder()` callable that returns an sklearn-compatible
    estimator (with .fit and .predict methods).

    historical: pre-tournament match history DataFrame (optional).
    ref_date: reference date for time-decay (defaults to 2026-06-24).
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

        row = played.loc[idx]
        ts = compute_team_stats(
            played,
            exclude_idx=idx,
            historical=historical,
            ref_date=ref_date,
        )
        f1 = _row_features(row["team1"], row["team2"], ts, rankings)
        f2 = _row_features(row["team2"], row["team1"], ts, rankings)
        X_pred = pd.DataFrame([f1, f2], columns=FEATURE_COLS)

        preds = np.clip(model.predict(X_pred), 0, None)
        pred_g1.append(preds[0])
        pred_g2.append(preds[1])
        true_g1.append(row["goals1"])
        true_g2.append(row["goals2"])

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
