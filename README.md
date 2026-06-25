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

**Attack/Defence rates** are computed as a weighted blend:
- **Pre-tournament history (2020–2026):** exponential time-decay (`weight = exp(−λ × days)`) multiplied by tournament relevance weight (World Cup × 3, Qualifiers × 2, Confederation × 1.5, Friendly × 0.5)
- **In-tournament group stage:** fixed weight equivalent to a World Cup match

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

- For each match, independent Poisson PMFs with estimated λ₁ and λ₂ are combined into a joint probability matrix P[i, j] = P(team1 scores i, team2 scores j)
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
- FIFA World Rankings methodology: https://www.fifa.com/fifa-world-ranking/procedure-men
