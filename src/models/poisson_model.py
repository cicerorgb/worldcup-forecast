"""
Poisson regression model for football score prediction.

Dixon & Coles (1997) showed that goals in football follow independent Poisson
distributions for each team, with λ depending on attack and defence strength
parameters. Here λ is estimated via Poisson GLM (log-link) with L2
regularization:

    log(λ) = β₀ + β₁·attack + β₂·defence_opp + β₃·rank + ...

Score probabilities are computed as products of independent Poisson PMFs.
A Dixon-Coles correction (ρ) is applied to low-scoring outcomes (0-0, 1-0,
0-1, 1-1), where the independence assumption is slightly violated.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(alpha: float = 0.3) -> Pipeline:
    """
    Pipeline: StandardScaler → PoissonRegressor.

    alpha: L2 regularisation strength (prevents overfitting on small datasets).
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", PoissonRegressor(alpha=alpha, max_iter=1000, fit_intercept=True)),
        ]
    )


def _dc_tau(g1: int, g2: int, l1: float, l2: float, rho: float) -> float:
    """Dixon-Coles correction factor for low-scoring outcomes."""
    if g1 == 0 and g2 == 0:
        return 1.0 - l1 * l2 * rho
    if g1 == 1 and g2 == 0:
        return 1.0 + l2 * rho
    if g1 == 0 and g2 == 1:
        return 1.0 + l1 * rho
    if g1 == 1 and g2 == 1:
        return 1.0 - rho
    return 1.0


def estimate_rho(played: "pd.DataFrame", model, X_train: "pd.DataFrame") -> float:
    """
    Estimates the Dixon-Coles ρ parameter via maximum likelihood,
    holding the trained model's λ predictions fixed.
    """
    lambdas = model.predict(X_train)

    def neg_log_lik(rho: float) -> float:
        ll = 0.0
        for i, (_, row) in enumerate(played.iterrows()):
            g1, g2 = int(row["goals1"]), int(row["goals2"])
            l1, l2 = float(lambdas[2 * i]), float(lambdas[2 * i + 1])
            tau = _dc_tau(g1, g2, l1, l2, rho)
            if tau <= 0:
                return 1e10
            ll += np.log(tau) + poisson.logpmf(g1, l1) + poisson.logpmf(g2, l2)
        return -ll

    result = minimize_scalar(neg_log_lik, bounds=(-0.2, 0.2), method="bounded")
    return float(result.x)


def score_probability_matrix(
    l1: float,
    l2: float,
    rho: float = 0.0,
    max_goals: int = 10,
) -> np.ndarray:
    """
    Returns matrix P[i, j] = P(team1 scores i, team2 scores j).
    Applies Dixon-Coles correction for i, j ∈ {0, 1}.
    """
    p1 = poisson.pmf(np.arange(max_goals + 1), l1)
    p2 = poisson.pmf(np.arange(max_goals + 1), l2)
    matrix = np.outer(p1, p2)

    for g1 in range(2):
        for g2 in range(2):
            matrix[g1, g2] *= _dc_tau(g1, g2, l1, l2, rho)

    matrix = np.clip(matrix, 0, None)
    matrix /= matrix.sum()
    return matrix


def most_likely_score(
    l1: float,
    l2: float,
    rho: float = 0.0,
    max_goals: int = 10,
) -> tuple[int, int, float]:
    """Returns (goals_team1, goals_team2, probability) of the most likely score."""
    matrix = score_probability_matrix(l1, l2, rho, max_goals)
    idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    return int(idx[0]), int(idx[1]), float(matrix[idx])


def result_probabilities(
    l1: float,
    l2: float,
    rho: float = 0.0,
    max_goals: int = 15,
) -> tuple[float, float, float]:
    """Returns (p_win_team1, p_draw, p_win_team2) marginalised from the score matrix."""
    matrix = score_probability_matrix(l1, l2, rho, max_goals)
    p_win1 = float(np.sum(np.tril(matrix, k=-1)))
    p_draw = float(np.sum(np.diag(matrix)))
    p_win2 = float(np.sum(np.triu(matrix, k=1)))
    return p_win1, p_draw, p_win2
