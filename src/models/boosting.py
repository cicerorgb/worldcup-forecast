"""
Modelos de Gradient Boosting para previsão de gols.

XGBoost e LightGBM com objetivo Poisson (log-link nativo), que é teoricamente
correto para contagem de gols. Hiperparâmetros conservadores para evitar
overfitting com o dataset pequeno (96 amostras = 48 jogos × 2 perspectivas).
"""

from __future__ import annotations

import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin


def build_xgb(random_state: int = 42) -> xgb.XGBRegressor:
    """
    XGBoost com objetivo Poisson.
    Regularização forte (max_depth=3, reg_lambda=2) para dataset pequeno.
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
    LightGBM com objetivo Poisson.
    num_leaves=8 e min_child_samples=6 evitam overfitting agressivo.
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
    Combina N modelos com pesos calculados a partir do MAE de LOO-CV.
    Peso do modelo i = (1/MAE_i) / soma(1/MAE_j)
    """

    def __init__(self, models: list, weights: list[float]):
        self.models = models
        import numpy as np
        w = np.array(weights, dtype=float)
        self.weights = w / w.sum()

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X):
        import numpy as np
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds @ self.weights
