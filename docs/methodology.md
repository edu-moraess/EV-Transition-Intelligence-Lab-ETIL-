# Metodologia

## Modelos de Difusão
- Bass Diffusion
- Logistic Growth
- Gompertz

## Detecção de Regimes
- Hidden Markov Models (HMM)
- Bayesian Change Point

## Previsão
- Gaussian Process Regression
- Monte Carlo Simulation

## EVQTI
- PCA + Z-score + Entropy Weighting


## Temporal availability contract

ETIL uses an annual panel with an explicit one-year feature lag and a one-year-ahead target:

- A panel row indexed by year `t` uses predictor values observed in year `t-1`.
- `growth_t1` on row `t` is `growth_rate` observed in year `t+1`.
- A forecast origin at year `T` may use a calibration row `t` only when its target observation year `t+1` is no later than `T`.
- Therefore, for an origin at the latest panel year `T`, the default FactorEngine calibration row cutoff is `T-1`. The parameter `calibration_end_year` denotes the last eligible *panel row year*, not the forecast-origin year.
- The score for row `T` is generated from lagged features and does not use `growth_t1` from that row. That target is an outcome for later evaluation only.

FactorEngine's expanding-origin evaluation truncates the model input at each origin year, fits calibration using only labels observable by that origin, generates the score, and joins the subsequent raw outcome only after scoring. EVQTI's feature and target winsorization bounds are calculated separately at each row year using earlier rows only.

**Vintage limitation:** annual observation years are not publication timestamps. Historical IEA releases may revise prior observations, and the repository does not currently contain point-in-time release vintages for every historical year. Walk-forward results therefore test temporal ordering within the available panel, but must not be described as a fully vintage-aware real-time backtest until release-date and revision history are available.
