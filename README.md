# World Cup 2026 — Score Forecaster

A machine learning pipeline for predicting match scores in the **2026 FIFA World Cup Group Stage**, combining Poisson regression, gradient boosting (XGBoost + LightGBM), and Dixon-Coles correction.

## Methodology

### Feature Engineering

Each match generates **two training rows** (one per attacking perspective), keeping the prediction symmetric. Features capture:

| Feature | Description |
|---|---|
| `attack_off` | Weighted goal-scoring rate of the attacking team |
| `defense_def` | Weighted goals-conceded rate of the defending team |
| `rank_off` / `rank_def` | FIFA ranking points (÷1000), proxy for team quality |
| `pts_off` / `pts_def` | Group-stage points per game |
| `rank_diff` | Normalised ranking gap |
| `atk_vs_def` | Attack rate minus opponent defence rate |
| `quality_ratio` | Ratio of FIFA ranking points |

**Attack/Defence rates** are computed as a weighted blend of two signals:

| Signal | Weight |
|---|---|
| Pre-tournament history (2020–2026) | 6-month bucket weight × tournament relevance weight |
| In-tournament group stage | Fixed weight = 3.0 (World Cup level) |

**6-month stepped decay** — inspired by the FIFA ranking methodology used before continuous Elo was adopted (Hvattum & Arntzen, 2010). Each historical result is assigned a bucket weight based on how long ago it occurred:

| Window | Bucket weight |
|---|---|
| 0–6 months | 1.00 (full weight) |
| 6–12 months | 0.70 |
| 12–18 months | 0.50 |
| 18–24 months | 0.35 |
| 24–30 months | 0.25 |
| 30+ months | 0.15 |

Calibrated to approximate an exponential curve with a ~12-month half-life (ξ ≈ 0.0019/day, standard Dixon-Coles practitioner range). Each bucket weight is then multiplied by the tournament relevance weight (World Cup × 3, Qualifiers × 2, Confederation × 1.5, Friendly × 0.5).

**Asymmetric loss penalty** — when a team *lost* a historical match, the goals conceded in that game receive an extra recency multiplier before being averaged into the defence rate. This reflects the intuition that a recent heavy defeat is a stronger signal of current defensive weakness than an old one:

| Window | Extra multiplier on goals conceded (losses only) |
|---|---|
| 0–6 months | × 1.50 |
| 6–12 months | × 1.25 |
| 12–18 months | × 1.10 |
| 18+ months | × 1.00 (no extra penalty) |

When no history is available, Bayesian smoothing towards a league-average prior is used as a fallback.

### Models

| Model | Implementation | Objective |
|---|---|---|
| Poisson GLM | `sklearn.PoissonRegressor` (L2 reg.) | Log-link Poisson |
| XGBoost | `xgb.XGBRegressor` | `count:poisson` |
| LightGBM | `lgb.LGBMRegressor` | `poisson` |
| **Weighted Ensemble** | Inverse-MAE weights from LOO-CV | Combination of all three |

### Evaluation

**Leave-One-Out Cross-Validation (LOO-CV)** is used over k-fold because the dataset is small (48 group-stage matches). Each fold excludes a single match from both team-statistics computation and the training set, preventing any form of data leakage.

Metrics reported: MAE, RMSE, W/D/L result accuracy, exact goal count accuracy, exact score accuracy.

### Score Probability Distribution

The model outputs **Dixon-Coles corrected score probability matrices**:

- For each match, independent Poisson PMFs with estimated $λ_1$ and $λ_2$ are combined into a joint probability matrix $P[i, j] = P(team_1\ scores_i,\ team_2\ scores_j)$
- A Dixon-Coles correction factor ρ is applied to low-scoring outcomes (0-0, 1-0, 0-1, 1-1), where the independence assumption is slightly violated
- ρ is estimated via maximum likelihood on the training data

## Project Structure

```
worldcup-forecast/
├── data/
│   ├── fase-de-grupos.CSV      # Group stage schedule and results
│   ├── pais-rank.CSV           # FIFA ranking points (48 nations)
│   └── *.html                  # Match history scraped from FIFA website
├── src/
│   ├── data_loader.py          # CSV ingestion and normalisation
│   ├── html_parser.py          # FIFA HTML match history parser
│   ├── features.py             # Feature engineering pipeline
│   ├── evaluation.py           # LOO-CV harness and metrics
│   └── models/
│       ├── poisson_model.py    # Poisson GLM + Dixon-Coles correction
│       └── boosting.py         # XGBoost, LightGBM, weighted ensemble
├── outputs/
│   ├── predictions.csv         # Predicted scores and probabilities
│   └── charts/                 # Generated visualisations
├── main.py                     # Pipeline entry point
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

The pipeline prints a progress log and a formatted prediction table to stdout, then writes:

| Output | Description |
|---|---|
| `outputs/predictions.csv` | Predicted scores, λ values, W/D/L probabilities, exact score probabilities |
| `outputs/charts/feature_importance.png` | XGBoost feature importance (gain) |
| `outputs/charts/score_heatmaps.png` | Score probability matrices for selected matches |
| `outputs/charts/result_probabilities.png` | Stacked bar chart of W/D/L probabilities |
| `outputs/charts/model_comparison.png` | LOO-CV performance with vs. without historical data |

## Output Schema

`predictions.csv` columns:

| Column | Description |
|---|---|
| `Date` | Match date (DD/MM) |
| `Group` | Group letter |
| `Team1` / `Team2` | Team names |
| `Predicted_Goals_T1` / `Predicted_Goals_T2` | Most likely score |
| `Lambda_T1` / `Lambda_T2` | Expected goals (Poisson rate parameters) |
| `P_Win_T1` / `P_Draw` / `P_Win_T2` | Win/Draw/Loss probabilities (%) |
| `P_Exact_Score` | Probability of the predicted exact score (%) |
| `Predicted_Result` | Textual result label |

## References

- Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores and inefficiencies in the football betting market. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 46(2), 265-280.
- Hvattum, L. M., & Arntzen, H. (2010). Using ELO ratings for match result prediction in association football. *International Journal of Forecasting*, 26(3), 460-470.
- FIFA World Rankings methodology: https://www.fifa.com/fifa-world-ranking/procedure-men
