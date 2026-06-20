# ⚡ EV Transition Intelligence Lab (ETIL)

Institutional quantitative platform for global EV transition analysis.

**Live app:** [quantev.streamlit.app](https://quantev.streamlit.app)  
**Author:** [Carlos Eduardo](https://www.linkedin.com/in/carlos-eduardo-2753a92b5) · [GitHub](https://github.com/edu-moraess)  
**Data sources:** IEA Global EV Outlook 2025/2026 · IEA Energy Prices 2026 · NewsAPI

---

## What this is

ETIL tracks the global shift to electric vehicles across 60+ countries — where it's happening, how fast, and what the data says about where it's going next.

The platform combines diffusion models, regime detection, quantitative factor analysis and a live news digest into a single research environment. It is designed to be both analytically rigorous and readable by anyone with an interest in energy transition.

---

## Pages

| Page | What it answers |
|---|---|
| 🌍 **Global Monitor** | Where is EV adoption happening and who is growing fastest? |
| 🔬 **Country Lab** | Deep-dive into any country — history, diffusion curve, fuel price context and adoption regime |
| 🧠 **Diffusion Engine** | Where is a country on the S-curve? Bass, Logistic and Gompertz models with R², AIC, BIC |
| 🔄 **Regime Detection** | Is a market Emerging, Expanding, Accelerating or Maturing? GMM + Bayesian Changepoint |
| 📊 **Forecast Lab** | Probabilistic 10-year forecast with 80/95% confidence intervals via Gaussian Process + Monte Carlo |
| 🏆 **EVQTI Ranking** | Which early-stage markets are most likely to grow fastest next year? Full statistical validation |
| 📰 **EV Pulse** | Live news digest with AI-generated analytical summaries (requires NewsAPI key) |

---

## Production results — EVQTI v3 (IEA 2026, Cars)

| Metric | Value | What it means |
|---|---|---|
| Mean Rank IC | **+0.237** | Strong predictive signal — threshold for "robust" is 0.10 |
| Information Ratio | **0.99** | Consistent signal across time — IR > 0.5 is the target |
| Years with positive IC | **93%** | Signal points in the right direction in 13 out of 14 years |
| Q5-Q1 growth spread | **+122pp** | Top-quintile countries grew 122pp faster than bottom-quintile |
| Quintile monotonicity | ✅ | Q1=49% → Q2=87% → Q3=106% → Q4=122% → Q5=171% |

---

## Project structure

```
├── app.py                          # Entry point — routing, data pipeline, dataset fallback
├── requirements.txt
├── .gitignore
│
├── .streamlit/
│   ├── config.toml                 # Light theme, navy #1f3a6e
│   └── secrets.toml                # API keys — never commit (see EV Pulse setup)
│
├── data/
│   ├── fuel_loader.py              # IEA Energy Prices — 33 countries, cached, fault-tolerant
│   └── raw/
│       ├── EV_Data_Explorer_2025_GEVO_clean.csv
│       ├── EV_Data_Explorer_2026_GEVO_clean.csv
│       └── fuel_prices_iea.csv
│
├── ingestion/
│   ├── loader.py                   # IEALoader — validates, loads, delegates to Aggregator
│   ├── aggregator.py               # Correct dimensional aggregation, no double-counting
│   └── dataset_engine.py           # Legacy compatibility layer
│
├── preprocessing/
│   ├── cleaning.py
│   ├── transformation.py
│   └── features.py
│
├── models/
│   ├── diffusion/                  # Bass · Logistic · Gompertz
│   ├── forecasting/                # Gaussian Process + Monte Carlo (80/95% CI)
│   ├── regimes/                    # GMM + Bayesian Changepoint
│   └── scoring/
│       └── evqti.py                # EVQTI v3 — IC-calibrated, anti-leakage
│
├── analytics/
│   ├── country_analysis.py
│   ├── global_ranking.py
│   └── scenario_engine.py
│
├── visualization/
│   └── theme.py                    # Centralised palette — dark/light mode safe
│
└── streamlit_pages/
    ├── 01_Global_Monitor.py
    ├── 02_Country_Lab.py
    ├── 03_Diffusion_Engine.py
    ├── 04_Regime_Detection.py
    ├── 05_Forecast_Lab.py
    ├── 06_EVQTI_Ranking.py
    └── 07_EV_Pulse.py              # NewsAPI + Claude AI summaries, cached and fault-tolerant
```

---

## EVQTI v3 — EV Quant Transition Index

### The intuition

Norway and the Netherlands are global EV leaders. But they rank *low* on EVQTI. Why?

Because their markets are maturing. When 35%+ of all cars on the road are already electric, growth naturally slows — there are fewer internal combustion car owners left to convert.

Countries like Brazil, Indonesia and South Africa rank *high* on EVQTI. They have low penetration today but are already accelerating. Early-stage markets grow faster. This is Bass diffusion law applied to the country cross-section.

### The maths

Signal built at t-1. Target: EV growth at t+1. No future information leaks in.

| Feature (t-1) | Direction | Weight | Empirical IC |
|---|---|---|---|
| EV stock share | Negative — lower = earlier stage | **48.5%** | −0.259 (p<0.001) |
| EV sales share | Negative — lower = earlier stage | **51.5%** | −0.275 (p<0.001) |

Growth rate was tested and excluded: IC = −0.037, p = 0.36 — no cross-country predictive power.

### Normalisation pipeline

1. Winsorise features at 1%–99%
2. Create lags (t-1) on winsorised values
3. Re-winsorise lags at their own P1/P99
4. Z-score cross-section per year
5. Score = −(w₁·z₁ + w₂·z₂)
6. Winsorise target at 5%–95%

### Validation

| Metric | Formula | Result |
|---|---|---|
| Rank IC | Spearman(EVQTI, growth_t+1) | +0.237 |
| t-stat | IC · √(n−2) / √(1−IC²) | Significant in 36% of years |
| IR | IC_mean / σ(IC) | 0.99 |
| Spread Q5-Q1 | Δ growth between quintiles | +122pp |

---

## Aggregation rules — no double-counting

IEA provides EV sales at two levels. Never mix them.

```
IEA 2026 has:  powertrain = 'EV'            (aggregate: BEV + PHEV + FCEV)
               powertrain = 'BEV'/'PHEV'... (individual components)

WRONG:  Total = EV + BEV + PHEV   ← double-counts
RIGHT:  Total = EV                ← use aggregate only
   or:  Total = BEV + PHEV + FCEV ← use components only

IEA 2025 has no 'EV' aggregate in EV sales → use BEV + PHEV + FCEV.
```

Primary key: `country × category × year × parameter × mode × powertrain`

---

## Fuel price context

IEA Energy Prices Monthly Excerpt (Jun 2026) — end-user prices including taxes.

| Coverage | 33 countries · Jan 2015 – May 2026 · USD/litre |
|---|---|
| Products | Gasoline · Diesel |
| Exclusions | Ukraine (insufficient data coverage) |

Available in Country Lab for all 33 covered countries as a dual-axis chart overlaying monthly fuel prices with annual EV sales.

**Fault tolerance:** the loader validates required columns, parses dates defensively (`errors="coerce"`), and returns an empty DataFrame on any malformed input rather than crashing the page. Cached for 1 hour via `@st.cache_data`.

---

## EV Pulse setup

EV Pulse requires a free NewsAPI key (100 requests/day).

**Step 1** — Get a key at [newsapi.org/register](https://newsapi.org/register)

**Step 2** — Add to `.streamlit/secrets.toml`:
```toml
NEWSAPI_KEY = "your_key_here"
ANTHROPIC_API_KEY = "your_key_here"   # optional — enables AI summaries
```

**Step 3** — On Streamlit Cloud: App Settings → Secrets → paste the same content.

**Resilience:** news results are cached for 1 hour and AI summaries for 24 hours, protecting the free-tier quota. Every failure mode (timeout, invalid key, rate limit, connection error) surfaces a specific, human-readable message instead of a generic error or a blank page. If only the Anthropic key is missing, headlines still display — AI summaries degrade gracefully rather than blocking the page.

---

## Multi-dataset

| Mode | Countries | Coverage | IEA schema |
|---|---|---|---|
| IEA 2026 | 63 | 2010–2025 | powertrain = `EV` (aggregate) |
| IEA 2025 | 54 | 2010–2024 | powertrain = `BEV / PHEV / FCEV` |
| Combined | 63 | 2010–2025 | 2026 takes priority on overlaps |

**Automatic fallback:** if the selected dataset fails to load (missing or corrupted file), `app.py` automatically falls back to the other available year and notifies the user with an inline message — a partial dataset is always preferred over a blank page.

---

## Local setup

```bash
git clone https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-
cd EV-Transition-Intelligence-Lab-ETIL-
pip install -r requirements.txt

# Add your API keys (see EV Pulse setup above)
# Edit .streamlit/secrets.toml with your keys

streamlit run app.py
```

---

## Dependencies

```
streamlit>=1.28.0    pandas>=2.2.0       numpy>=1.26.0
scipy>=1.11.0        scikit-learn>=1.3.0  plotly>=5.18.0
statsmodels>=0.14.0  openpyxl>=3.1.0     requests>=2.31.0
```

`hmmlearn` removed — incompatible with Python 3.14. Regime detection uses `GaussianMixture` (scikit-learn).

---

## License

MIT License — see [LICENSE](LICENSE).  
IEA data © OECD/IEA. Used for non-commercial research only.  
EVQTI is a research construct, not financial advice.

---

**Carlos Eduardo** · Quant Data Scientist · Edumetria  
[LinkedIn](https://www.linkedin.com/in/carlos-eduardo-2753a92b5) · [GitHub](https://github.com/edu-moraess) · [App](https://quantev.streamlit.app)

*IEA Global EV Outlook 2025/2026 · IEA Energy Prices 2026 · Models: GMM · GP · Bass · Monte Carlo · EVQTI*
