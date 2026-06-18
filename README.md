# ⚡ EV Transition Intelligence Lab (ETIL)

Institutional quantitative platform for global EV transition analysis.

**Data source:** IEA Global EV Outlook 2025 · 2026 · IEA Energy Prices 2026  
**Deploy:** [quantev.streamlit.app](https://quantev.streamlit.app)  
**Author:** [Carlos Eduardo](https://www.linkedin.com/in/carlos-eduardo-2753a92b5) · [GitHub](https://github.com/edu-moraess)

---

## Overview

ETIL combines technology diffusion models, regime detection and a proprietary quantitative factor — the **EVQTI** — to map the EV transition stage of 60+ countries with institutional-grade statistical rigour.

Core premise: **future growth concentrates in markets that have not yet arrived, not in those that already have.** The EVQTI operationalises Bass diffusion law into a measurable, anti-leakage, empirically validated factor.

---

## Production results — EVQTI v3 (IEA 2026, Cars)

| Metric | Value | Interpretation |
|---|---|---|
| Mean Rank IC | **+0.237** | Robust signal — utility threshold is 0.10 |
| Information Ratio (IR) | **0.99** | IR > 0.5 = consistent over time |
| % years IC > 0 | **93%** | Signal fails in fewer than 1 in 14 years |
| Q5-Q1 spread | **+122pp** | Growth difference between top and bottom quintile |
| Monotonic quintiles | ✅ | Q1=49% → Q2=87% → Q3=106% → Q4=122% → Q5=171% |

---

## Project structure

```
├── app.py                        # Entry point — routing and pipeline
├── requirements.txt
├── .streamlit/
│   └── config.toml               # Light theme — navy #1f3a6e
│
├── data/
│   ├── fuel_loader.py            # IEA Energy Prices loader (33 countries)
│   └── raw/
│       ├── EV_Data_Explorer_2025_GEVO_clean.csv
│       ├── EV_Data_Explorer_2026_GEVO_clean.csv
│       └── fuel_prices_iea.csv
│
├── ingestion/
│   ├── dataset_engine.py         # Multi-dataset: 2025 / 2026 / combined
│   └── loader.py
│
├── preprocessing/
│   ├── cleaning.py
│   ├── transformation.py
│   └── features.py
│
├── models/
│   ├── diffusion/                # Bass · Logistic · Gompertz (with R²)
│   ├── forecasting/              # Gaussian Process + Monte Carlo (80/95% CI)
│   ├── regimes/                  # GMM + Bayesian Changepoint
│   └── scoring/
│       └── evqti.py              # EVQTI v3 — IC-calibrated weights
│
├── analytics/
│   ├── country_analysis.py
│   ├── global_ranking.py
│   └── scenario_engine.py
│
├── visualization/
│   └── theme.py                  # Centralised quant palette — dark/light safe
│
└── streamlit_pages/
    ├── 01_Global_Monitor.py
    ├── 02_Country_Lab.py
    ├── 03_Diffusion_Engine.py
    ├── 04_Regime_Detection.py
    ├── 05_Forecast_Lab.py
    └── 06_EVQTI_Ranking.py
```

---

## Pages

| Page | Description |
|---|---|
| 🌍 Global Monitor | EV stock share map, global KPIs, time series, top 10 countries |
| 🔬 Country Lab | EV history, YoY, stock share, diffusion models, regimes, **fuel price context** |
| 🧠 Diffusion Engine | Bass / Logistic / Gompertz with R², AIC, BIC, multi-country comparison |
| 🔄 Regime Detection | GMM regime probabilities + Bayesian Changepoint |
| 📊 Forecast Lab | Gaussian Process + Monte Carlo with 80/95% CI, percentile table |
| 🏆 EVQTI Ranking | IC, Rank IC, t-stat, IR, quintile analysis, full ranking |

---

## EVQTI v3 — EV Quant Transition Index

### Economic rationale

Countries in the early stage of EV diffusion — low current penetration with latent demand — concentrate the highest future growth. This is Bass diffusion law applied to the country cross-section: saturated markets grow slower; emerging markets grow faster.

### Factor construction (anti-leakage)

Signal computed with lagged features (t-1). Target is EV growth at t+1.

| Feature (t-1) | Direction | Weight | Empirical IC |
|---|---|---|---|
| ev_stock_share | Negative (early-stage) | **48.5%** | −0.259 (p<0.001) |
| ev_sales_share | Negative (early-stage) | **51.5%** | −0.275 (p<0.001) |

> `growth_rate_lag1` excluded: IC = −0.037, p = 0.36 — not statistically significant.

### Normalisation pipeline

1. Winsorise features at 1%–99%
2. Create lags (t-1) on winsorised values
3. Re-winsorise lags at their own P1/P99
4. Cross-section z-score per year
5. Score = −(w₁·z₁ + w₂·z₂)
6. Winsorise target at 5%–95%

### Validation metrics

| Metric | Formula | Threshold |
|---|---|---|
| Rank IC | Spearman(EVQTI, growth_t+1) | > 0.10 = robust |
| t-stat | IC · √(n−2) / √(1−IC²) | — |
| IR | IC_mean / σ(IC) | > 0.5 = consistent |
| Spread Q5-Q1 | Δ growth between quintiles | — |

---

## Fuel price context (Country Lab)

Integrated for 33 countries via IEA Energy Prices Monthly Excerpt (Jun 2026).

| Metric | Description |
|---|---|
| Gasoline price | Latest USD/litre (end-user, taxes included) |
| Diesel price | Latest USD/litre |
| Gasoline YoY | Year-on-year price change (%) |
| Fuel volatility | Annualised std dev of monthly gasoline returns (last 12 months) |

Displayed as a dual-axis chart overlaying monthly fuel prices with annual EV sales.  
Ukraine excluded: insufficient data coverage.

---

## Multi-dataset

| Mode | Countries | Coverage | IEA schema |
|---|---|---|---|
| IEA 2026 | 63 | 2010–2025 | powertrain = `EV` (aggregated) |
| IEA 2025 | 54 | 2010–2024 | powertrain = `BEV / PHEV / FCEV` |
| Combined | 63 | 2010–2025 | 2026 takes priority on overlaps |

Schema auto-detected via presence of `EV` in `EV sales` records.

---

## Design system

All colours defined in `visualization/theme.py` and imported by all pages.  
No hex codes hardcoded in individual page files.  
Legend and font colours are transparent / inherited — compatible with both light and dark mode.

| Element | Colour | Use |
|---|---|---|
| Primary / positive | `#1f3a6e` navy | Bars, main lines, ranking |
| Negative / alert | `#8b1a1a` dark red | Negative IC, negative growth |
| Momentum | `#b7770d` amber | Changepoints, moderate alerts |
| Positive soft | `#1a4731` dark green | Growth, acceleration regimes |
| Secondary text | `#4a4a4a` slate | Labels, annotations |

---

## Local setup

```bash
git clone https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-
cd EV-Transition-Intelligence-Lab-ETIL-
pip install -r requirements.txt
streamlit run app.py
```

Data files must be placed in `data/raw/`.

---

## Dependencies

```
streamlit>=1.28.0   pandas>=2.2.0       numpy>=1.26.0
scipy>=1.11.0       scikit-learn>=1.3.0  plotly>=5.18.0
statsmodels>=0.14.0 openpyxl>=3.1.0
```

> `hmmlearn` removed — incompatible with Python 3.14.  
> Regime detection uses `GaussianMixture` (scikit-learn) with equivalent interface.

---

## Author

**Carlos Eduardo**  
Quant Data Scientist · Edumetria

[LinkedIn](https://www.linkedin.com/in/carlos-eduardo-2753a92b5) · [GitHub](https://github.com/edu-moraess) · [App](https://quantev.streamlit.app)

---

*Source: IEA Global EV Outlook 2025/2026 · IEA Energy Prices 2026*  
*Models: GMM · GP · Bass · Monte Carlo · EVQTI*
