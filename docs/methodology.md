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

- A panel row indexed by year `t` uses predictor values observed in calendar year `t-1`; if that exact year is absent, the lag is missing.
- `growth_t1` on row `t` is `growth_rate` observed in calendar year `t+1`; if that exact year is absent, the target is missing.
- A forecast origin at year `T` may use a calibration row `t` only when its target observation year `t+1` is no later than `T`.
- Therefore, for an origin at the latest panel year `T`, the default FactorEngine calibration row cutoff is `T-1`. The parameter `calibration_end_year` denotes the last eligible *panel row year*, not the forecast-origin year.
- The score for row `T` is generated from lagged features and does not use `growth_t1` from that row. That target is an outcome for later evaluation only.

FactorEngine's expanding-origin evaluation truncates the model input at each origin year, fits calibration using only labels observable by that origin, generates the score, and joins the subsequent raw outcome only after scoring. EVQTI's feature and target winsorization bounds are calculated separately at each row year using earlier rows only. Feature engineering and both scoring engines require exact calendar adjacency; a missing annual observation is never silently treated as the previous or next year.

**Vintage limitation:** annual observation years are not publication timestamps. Historical IEA releases may revise prior observations, and the repository does not currently contain point-in-time release vintages for every historical year. Walk-forward results therefore test temporal ordering within the available panel, but must not be described as a fully vintage-aware real-time backtest until release-date and revision history are available.


## EVQTI — walk-forward calibration

The current country ranking uses fixed reference weights (0.485 for EV stock
share and 0.515 for EV sales share). These weights were derived from the full
historical IEA 2026 Cars panel (2010–2025); they are descriptive reference
weights and must not be presented as point-in-time or independently
out-of-sample calibrated.

The EVQTI `backtest()` uses a separate rolling-origin evaluation:

1. At test origin year `T`, the training window contains the preceding
   `train_years` calendar years, ending at `T-1`.
2. A training row `t` is eligible only when its next-year label
   `growth_t1` is observable by origin `T` (therefore `t < T`).
3. For each feature, directional annual Spearman ICs are calculated inside
   the training window. Positive mean directional ICs determine normalized
   weights; if no positive directional signal is supported, equal weights
   are used and explicitly marked as a fallback.
4. The score at `T` is evaluated against raw, unclipped growth at `T+1`.
   The test outcome is not used to estimate the weights.
5. The returned Rank IC, Pearson IC, top-minus-bottom quintile spread,
   sample size, calibration years and weights are recorded by test window.

This protocol reduces full-sample weight-selection leakage. It does not solve
historical IEA publication/revision vintages: observation year is not the
same as the date on which a value first became available. Results must
therefore be described as train-only temporal walk-forward validation over
the available release, not as a fully vintage-aware real-time backtest.
The fixed-weight current ranking and its historical IC/quintile summaries
remain in-sample reference diagnostics.
