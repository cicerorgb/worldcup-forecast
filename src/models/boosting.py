"""
Gradient Boosting models for goal-count prediction.

XGBoost and LightGBM with a Poisson objective (native log-link), which is
theoretically correct for count data. Hyper-parameters are conservative to
avoid overfitting on the small dataset (96 samples = 48 matches × 2 team
perspectives).
"""

from __future__ import annotations

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin


def build_xgb(random_state: int = 42) -> xgb.XGBRegressor:
    """
    XGBoost with Poisson objective.
    Strong regularization (max_depth=3, reg_lambda=2) suited for small datasets.
    """
    return xgb.XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.04,
        min_child_weight=4,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=2.0,
        objective="count:poisson",
        random_state=random_state,
        verbosity=0,
    )


def build_lgb(random_state: int = 42) -> lgb.LGBMRegressor:
    """
    LightGBM with Poisson objective.
    num_leaves=8 and min_child_samples=6 prevent aggressive overfitting.
    """
    return lgb.LGBMRegressor(
        n_estimators=300,
        num_leaves=8,
        learning_rate=0.04,
        min_child_samples=6,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=2.0,
        objective="poisson",
        random_state=random_state,
        verbose=-1,
    )


class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """
    Combines N estimators using inverse-MAE weights from LOO-CV.

    Weight of model i = (1 / MAE_i) / sum(1 / MAE_j)
    """

    def __init__(self, models: list, weights: list[float]):
        self.models = models
        w = np.array(weights, dtype=float)
        self.weights = w / w.sum()

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        preds = np.column_stack([model.predict(X) for model in self.models])
        return preds @ self.weights
