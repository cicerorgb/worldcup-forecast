"""
Modelo de Poisson para previsão de gols em futebol.

Fundamentação:
  Dixon & Coles (1997) demonstraram que gols em futebol seguem distribuições
  de Poisson independentes para cada time, com λ dependente de parâmetros de
  ataque e defesa. Aqui, λ é estimado via Regressão de Poisson (GLM com log-link),
  uma abordagem moderna equivalente com regularização L2 nativa.

  O modelo prevê quantos gols um time marca como:
    log(λ) = β₀ + β₁·attack + β₂·defense_opp + β₃·rank + ... (features)

  As probabilidades de placar são calculadas como produtos de PMFs Poisson
  independentes, e a correção de Dixon-Coles (ρ) é aplicada para os placares
  baixos (0-0, 1-0, 0-1, 1-1), onde a independência é ligeiramente violada.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(alpha: float = 0.3) -> Pipeline:
    """
    Pipeline: StandardScaler → PoissonRegressor.
    alpha: regularização L2 (evita overfitting com poucos dados).
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", PoissonRegressor(alpha=alpha, max_iter=1000, fit_intercept=True)),
        ]
    )


def _dc_tau(g1: int, g2: int, l1: float, l2: float, rho: float) -> float:
    """Fator de correção Dixon-Coles para placares baixos."""
    if g1 == 0 and g2 == 0:
        return 1.0 - l1 * l2 * rho
    if g1 == 1 and g2 == 0:
        return 1.0 + l2 * rho
    if g1 == 0 and g2 == 1:
        return 1.0 + l1 * rho
    if g1 == 1 and g2 == 1:
        return 1.0 - rho
    return 1.0


def estimate_rho(played, model, X_train: "pd.DataFrame") -> float:
    """
    Estima o parâmetro ρ de Dixon-Coles por maximum likelihood
    nos dados de treino, mantendo λ fixo do modelo treinado.
    """
    lambdas = model.predict(X_train)

    def neg_log_lik(rho: float) -> float:
        ll = 0.0
        for i, (_, row) in enumerate(played.iterrows()):
            g1, g2 = int(row["gols1"]), int(row["gols2"])
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
    Matriz P[i, j] = P(time1 marca i gols, time2 marca j gols).
    Inclui correção Dixon-Coles para i, j ∈ {0, 1}.
    """
    p1 = poisson.pmf(np.arange(max_goals + 1), l1)
    p2 = poisson.pmf(np.arange(max_goals + 1), l2)
    matrix = np.outer(p1, p2)

    for g1 in range(2):
        for g2 in range(2):
            tau = _dc_tau(g1, g2, l1, l2, rho)
            matrix[g1, g2] *= tau

    matrix = np.clip(matrix, 0, None)
    matrix /= matrix.sum()
    return matrix


def most_likely_score(
    l1: float,
    l2: float,
    rho: float = 0.0,
    max_goals: int = 10,
) -> tuple[int, int, float]:
    """Retorna (gols_time1, gols_time2, probabilidade) do placar mais provável."""
    matrix = score_probability_matrix(l1, l2, rho, max_goals)
    idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    return int(idx[0]), int(idx[1]), float(matrix[idx])


def result_probabilities(
    l1: float,
    l2: float,
    rho: float = 0.0,
    max_goals: int = 15,
) -> tuple[float, float, float]:
    """Retorna (p_vitória_time1, p_empate, p_vitória_time2)."""
    matrix = score_probability_matrix(l1, l2, rho, max_goals)
    p_win1 = float(np.sum(np.tril(matrix, k=-1)))
    p_draw = float(np.sum(np.diag(matrix)))
    p_win2 = float(np.sum(np.triu(matrix, k=1)))
    return p_win1, p_draw, p_win2
