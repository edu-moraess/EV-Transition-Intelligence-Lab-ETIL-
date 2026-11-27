# Metodologia

## Modelos de difusão

- **Bass Diffusion** — curva de difusão com parâmetros sujeitos a restrições de domínio.
- **Logistic Growth** — curva logística para aproximação de trajetórias cumulativas.
- **Gompertz** — curva assimétrica para trajetórias de adoção.

As métricas de ajuste calculadas sobre a amostra usada na estimação são diagnósticos *in-sample*. Não constituem, isoladamente, validação preditiva fora da amostra.

## Detecção e descrição de regimes

- **Gaussian Mixture Model (GMM)** — agrupamento probabilístico exploratório das observações. Os componentes são ordenados por uma feature de adoção para produzir rótulos descritivos.
- **Triagem heurística de mudança de média** — o módulo exposto pela classe legada `BayesianChangePoint` implementa um detector padronizado de deslocamento de média; não é um modelo bayesiano de change point.
- A classe legada `HMMRegimeDetector` usa GMM; apesar do nome histórico, não implementa um Hidden Markov Model.

Os regimes não devem ser interpretados como estados econômicos naturais, permanentes ou causalmente identificados. Os resultados dependem das features e da especificação escolhida.

## Previsão e simulação

- **Gaussian Process Regression** — modelo de regressão probabilística utilizado no Forecast Lab.
- **Monte Carlo Simulation** — geração de trajetórias condicionais às hipóteses e à estrutura de covariância do modelo.

Os quantis das trajetórias são quantis empíricos de simulação condicionados ao modelo e ao cenário. Não devem ser chamados automaticamente de intervalos de confiança ou credibilidade.

## EVQTI — EV Transition Quantitative Indicator

A especificação de referência utiliza duas features defasadas:

- `ev_stock_share_lag1`: participação de EV no estoque, defasada em um ano;
- `ev_sales_share_lag1`: participação de EV nas vendas, defasada em um ano.

A direção investigada é negativa: menor penetração defasada pode estar associada a maior crescimento subsequente. O score cross-sectional usa normalização z-score e pesos de referência fixos de 0,485 para stock share e 0,515 para sales share.

**Importante:** os pesos fixos foram derivados do painel histórico IEA 2026 Cars (2010–2025). São pesos descritivos de referência, não pesos point-in-time nem estimativas independentes out-of-sample.

A avaliação rolling-origin é um procedimento separado. Em cada origem, os pesos são estimados apenas com informação de treino anterior e labels observáveis até aquela origem. Se nenhum sinal direcional positivo for sustentado no treino, é aplicado fallback equal-weight explicitamente identificado. A avaliação utiliza o crescimento futuro bruto (`growth_t1`); a versão winsorizada é apenas diagnóstica.

## Contrato de disponibilidade temporal

ETIL uses an annual panel with an explicit one-year feature lag and a one-year-ahead target:

- A panel row indexed by year `t` uses predictor values observed in calendar year `t-1`; if that exact year is absent, the lag is missing.
- `growth_t1` on row `t` is `growth_rate` observed in calendar year `t+1`; if that exact year is absent, the target is missing.
- A forecast origin at year `T` may use a calibration row `t` only when its target observation year `t+1` is no later than `T`.
- Therefore, for an origin at the latest panel year `T`, the default FactorEngine calibration row cutoff is `T-1`. The parameter `calibration_end_year` denotes the last eligible *panel row year*, not the forecast-origin year.
- The score for row `T` is generated from lagged features and does not use `growth_t1` from that row. That target is an outcome for later evaluation only.

FactorEngine's expanding-origin evaluation truncates the model input at each origin year, fits calibration using only labels observable by that origin, generates the score, and joins the subsequent raw outcome only after scoring. EVQTI's feature and target winsorization bounds are calculated separately at each row year using earlier rows only. Feature engineering and both scoring engines require exact calendar adjacency; a missing annual observation is never silently treated as the previous or next year.

**Vintage limitation:** annual observation years are not publication timestamps. Historical IEA releases may revise prior observations, and the repository does not currently contain point-in-time release vintages for every historical year. Walk-forward results therefore test temporal ordering within the available panel, but must not be described as a fully vintage-aware real-time backtest until release-date and revision history are available.

## EVQTI — walk-forward calibration

The current country ranking uses fixed reference weights (0.485 for EV stock share and 0.515 for EV sales share). These weights were derived from the full historical IEA 2026 Cars panel (2010–2025); they are descriptive reference weights and must not be presented as point-in-time or independently out-of-sample calibrated.

The EVQTI `backtest()` uses a separate rolling-origin evaluation:

1. At test origin year `T`, the training window contains the preceding `train_years` calendar years, ending at `T-1`.
2. A training row `t` is eligible only when its next-year label `growth_t1` is observable by origin `T` (therefore, `t < T`).
3. For each feature, directional annual Spearman ICs are calculated inside the training window. Positive mean directional ICs determine normalized weights; if no positive directional signal is supported, equal weights are used and explicitly marked as a fallback.
4. The score at `T` is evaluated against raw, unclipped growth at `T+1`. The test outcome is not used to estimate the weights.
5. The returned Rank IC, Pearson IC, top-minus-bottom quintile spread, sample size, calibration years and weights are recorded by test window.

This protocol reduces full-sample weight-selection leakage. It does not solve historical IEA publication/revision vintages: observation year is not the same as the date on which a value first became available. Results must therefore be described as train-only temporal walk-forward validation over the available release, not as a fully vintage-aware real-time backtest. The fixed-weight current ranking and its historical IC/quintile summaries remain in-sample reference diagnostics.
