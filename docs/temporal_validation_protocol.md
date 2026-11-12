# ETIL — Temporal Validation and Scientific Integrity Protocol

**Status:** Proposed protocol baseline  
**Date:** 2026-11-12  
**Change type:** Documentation only; no runtime behavior changed.

## 1. Purpose

This protocol defines minimum evidence required before ETIL describes a score, ranking or forecast as out-of-sample predictive.

A chronological feature lag alone does not establish leakage-free validation. Every learned or data-adaptive operation must be fitted using information available at the forecast origin.

## 2. Forecast-origin information rule

For a forecast origin (t), every input to model development must be available at or before (t), subject to the actual publication/release date of the source.

This restriction applies to:

- Missing-value treatment and imputation.
- Winsorization bounds and outlier treatment.
- Scaling and normalization parameters.
- Feature selection.
- Correlation and information-coefficient estimation.
- Factor weights.
- Hyperparameter selection.
- Market-potential or ceiling estimation.
- Regime definitions and cluster-label interpretation.
- Threshold selection and calibration.
- Any data-dependent country or sample filtering.

A feature labeled (t-1) can still leak future information if its transformation or selection was estimated using the full panel.

## 3. Required evaluation design

Use expanding-window evaluation as the default:

1. Define an initial training period.
2. Select a forecast origin (t).
3. Fit all adaptive transformations and model parameters using eligible training data only.
4. Generate predictions or scores for the next period/horizon.
5. Store the prediction, origin, training window, source versions and fitted parameters.
6. Reveal the outcome only for evaluation.
7. Advance the origin and repeat.
8. Aggregate metrics only after all eligible origins have been evaluated.

Rolling windows may be used when there is a justified reason to limit historical memory. The window policy must be specified before evaluation.

## 4. EVQTI-specific requirements

EVQTI must distinguish two research questions:

### Fixed published signal persistence
A genuinely frozen signal specification and weights are fixed before the evaluation period. The evaluation tests whether that pre-existing signal continues to rank future outcomes. All preprocessing bounds and feature definitions must also be frozen using information available before the evaluation period.

### Recalibrated predictive signal
At each forecast origin, winsorization bounds, transformations, weights and any selection decisions are estimated using training data only. The resulting model scores the next period.

These are different estimands and must not be presented as the same backtest.

For each origin, record:
- Training years and eligible countries.
- Feature and target definitions.
- Calibration cutoff.
- Winsorization bounds.
- Feature coverage and missingness.
- Weight estimates and signs.
- Number of countries scored.
- Forecast horizon and realized target.
- Rank IC, Pearson IC and uncertainty/dispersion diagnostics.

Do not winsorize the evaluation target using quantiles calculated from the full historical panel. If target clipping is required for model fitting, estimate its thresholds from training targets only. Report unclipped outcome metrics as a robustness check.

## 5. FactorEngine-specific requirements

At each origin:

- Construct features with explicit as-of semantics.
- Fit feature transformations using training observations only.
- Estimate factor ICs and weights only from eligible training outcomes.
- Freeze weights before scoring the test period.
- Do not use test-period targets for factor selection, weights, thresholds or missingness decisions.
- Recompute the model independently at each origin.
- Record which factors were unavailable and how the score was renormalized.
- Report factor coverage and score coverage alongside rankings.

If a factor depends on an external series, align it by publication availability, not merely by observation year.

## 6. Baselines and metrics

At minimum compare against:

- Naive persistence / previous-period ranking.
- Simple univariate penetration baseline.
- Equal-weight factor baseline.
- Existing fixed-weight specification, where applicable.

For forecasts, choose metrics appropriate to the target and horizon, potentially including MAE, RMSE, MASE or sMAPE where mathematically suitable, plus interval coverage and a proper probabilistic score such as CRPS when predictive distributions are available.

For rankings, report:
- Spearman Rank IC by origin.
- Rank IC mean, dispersion and positive fraction.
- Rank stability.
- Top/bottom group spread with uncertainty.
- Country counts and coverage by origin.

Do not rely on a single pooled statistic that hides temporal instability.

## 7. Statistical cautions

- Cross-country observations are not necessarily independent.
- Small annual cross-sections make p-values and quintile spreads unstable.
- Multiple testing across years, features and model variants must be acknowledged.
- Country composition changes can alter rank metrics.
- Structural breaks and source-definition changes can invalidate apparent continuity.
- Missing-not-at-random reporting can bias cross-country comparisons.
- A good in-sample fit does not establish predictive skill.
- Empirical simulation quantiles are not automatically confidence or credible intervals.

Where sample size permits, use uncertainty estimates that respect country and time dependence. If the data do not support reliable inference, report descriptive diagnostics without overstating significance.

## 8. Reproducibility record

Every evaluation run should be identifiable by:

- Git commit SHA.
- Dataset file hashes and source releases.
- Data transformation version.
- Forecast origins and horizons.
- Training/test country-year keys.
- Feature schema.
- Model and hyperparameter configuration.
- Random seeds where applicable.
- Fitted preprocessing parameters.
- Metrics and prediction artifacts.
- Warnings, exclusions and failure reasons.

No result is considered reproducible if its input data and configuration cannot be reconstructed.

## 9. Acceptance gates

A model or index may be described as temporally validated only when:

1. Forecast origins and horizons are pre-specified.
2. All adaptive transformations are fitted within training windows.
3. Test targets are isolated from model development.
4. Baselines are evaluated on identical origins and samples.
5. Results are reported by origin and horizon.
6. Coverage, exclusions and missingness are visible.
7. Sensitivity analyses are documented.
8. Tests demonstrate that adding future observations cannot alter past fitted transformations or predictions.
9. The complete evaluation can be reproduced from a recorded commit and data versions.

Until these gates are met, label results as exploratory, retrospective or in-sample as appropriate.

## 10. Implementation order

1. Add tests for temporal leakage invariants.
2. Separate panel construction from train-fitted transformations.
3. Introduce explicit fit/calibrate/score APIs with cutoff validation.
4. Implement an expanding-window evaluation runner.
5. Add baseline and coverage reporting.
6. Persist run metadata and predictions.
7. Re-evaluate EVQTI and FactorEngine.
8. Only then extend these validated patterns to other ETIL domains.

**Principle:** Every score and forecast must be reproducible using only information that would have been available when it was produced.
