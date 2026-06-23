# ⚡ EV Transition Intelligence Lab (ETIL)

Institutional quantitative platform for global EV transition analysis.

**Live app:** [quantev.streamlit.app](https://quantev.streamlit.app)  
**Author:** [Carlos Eduardo](https://www.linkedin.com/in/carlos-eduardo-2753a92b5) · [GitHub](https://github.com/edu-moraess)  
**Data sources:** IEA Global EV Outlook 2025/2026 · IEA Energy Prices 2026 · World Bank API · NewsAPI

---

## What this is

ETIL tracks the global shift to electric vehicles across 63 countries — where it's happening, how fast, and what the data says about where it's going next.

The platform combines diffusion models, regime detection, infrastructure analysis, policy simulation and a proprietary two-factor composite index into a single research environment. It is designed to be analytically rigorous and readable by anyone interested in energy transition.

---

## Pages

| Page | What it answers |
|---|---|
| 🌍 **Global Monitor** | Where is EV adoption happening, who is growing fastest, what does the fleet look like? |
| 🔬 **Country Lab** | Full country deep-dive — sales, momentum, fuel context, World Bank fundamentals, diffusion models, adoption regime |
| 🧠 **Diffusion Engine** | Where is each country on the S-curve? Bass, Logistic and Gompertz with R², AIC, BIC — plus multi-country normalised comparison |
| 🔄 **Regime Detection** | Is a market Emerging, Expanding, Accelerating or Maturing? GMM + Bayesian Changepoint |
| 📊 **Forecast Lab** | Probabilistic 10-year forecast with 80/95% CI via Gaussian Process + Monte Carlo (2,000 paths) |
| 🏆 **EVQTI Ranking** | Which early-stage markets are most likely to grow fastest next year? Full IC/IR validation + rolling backtest |
| 📰 **EV Pulse** | Live news digest with AI-generated analytical summaries (requires NewsAPI key) |
| 🔌 **Infrastructure** | Charging network build-out: total chargers, EVs per charger vs IEA target, charger type mix, infrastructure vs sales race |
| 🎯 **Scenarios** | What if? — Bass diffusion scenarios with adjustable p, q, M parameters; uncertainty band across optimistic/conservative |
| 🌐 **ETII Composite Index** | Combines EVQTI momentum signal with World Bank structural capacity into a single 0–100 ranking with opportunity matrix |

---

## Production results — EVQTI v3 (IEA 2026, Cars)

| Metric | Value | What it means |
|---|---|---|
| Mean Rank IC | **+0.237** | Strong predictive signal — robust threshold is 0.10 |
| Information Ratio | **0.99** | Consistent signal across time — IR > 0.5 is the target |
| Years with positive IC | **93%** | Signal points in the right direction in 13 out of 14 years |
| Q5-Q1 growth spread | **+122pp** | Top-quintile countries grew 122pp faster than bottom-quintile |
| Quintile monotonicity | ✅ | Q1=49% → Q2=87% → Q3=106% → Q4=122% → Q5=171% |
| Rolling backtest IC | **+0.16 → +0.52** | Signal strengthened over time — not degrading |

---

## Project structure

```
├── app.py                          # Entry point — routing, pipeline, dataset fallback
├── requirements.txt
├── .gitignore
│
├── .streamlit/
│   ├── config.toml                 # Light theme, royal blue #1551d1
│   └── secrets.toml                # API keys — never commit
│
├── data/
│   ├── fuel_loader.py              # IEA Energy Prices — 33 countries, cached, fault-tolerant
│   ├── world_bank_client.py        # World Bank API — GDP, urbanisation, fundamentals score
│   └── raw/
│       ├── EV_Data_Explorer_2025_GEVO_clean.csv
│       ├── EV_Data_Explorer_2026_GEVO_clean.csv
│       └── fuel_prices_iea.csv
│
├── ingestion/
│   ├── loader.py                   # IEALoader — validates, loads, delegates to Aggregator
│   ├── aggregator.py               # Correct dimensional aggregation — no double-counting
│   │                               # charging_points(), charging_by_type(), ev_per_charger()
│   └── dataset_engine.py           # Legacy compatibility layer
│
├── preprocessing/
│   ├── cleaning.py
│   ├── transformation.py
│   └── features.py                 # Computes: cagr, growth_rate, momentum_3y, volatility,
│                                   #            ev_per_capita, ev_per_charger, penetration
│
├── models/
│   ├── diffusion/                  # Bass · Logistic · Gompertz
│   ├── forecasting/                # Gaussian Process + Monte Carlo (80/95% CI)
│   ├── regimes/                    # GMM + Bayesian Changepoint
│   └── scoring/
│       └── evqti.py                # EVQTI v3 with backtest() method
│
├── analytics/
│   ├── country_analysis.py
│   ├── global_ranking.py
│   ├── scenario_engine.py          # Bass scenario simulation (Baseline/Optimistic/Conservative)
│   └── scenario_engine.py
│
├── visualization/
│   └── theme.py                    # Centralised palette — vibrant, modebar hidden project-wide
│
├── streamlit_pages/
│   ├── 01_Global_Monitor.py
│   ├── 02_Country_Lab.py           # + momentum, World Bank fundamentals
│   ├── 03_Diffusion_Engine.py      # + multi-country S-curve normalised view + parameter table
│   ├── 04_Regime_Detection.py
│   ├── 05_Forecast_Lab.py
│   ├── 06_EVQTI_Ranking.py         # + rolling backtest chart
│   ├── 07_EV_Pulse.py              # NewsAPI + Claude AI summaries
│   ├── 08_Infrastructure.py        # NEW — charging network intelligence
│   ├── 09_Scenarios.py             # NEW — Bass policy scenario engine
│   └── 10_ETII.py                  # NEW — composite index (EVQTI + Fundamentals)
│
├── tests/
│   ├── test_aggregator.py          # 29 regression tests (double-counting, schema detection)
│   ├── test_loader.py
│   └── test_evqti.py               # Anti-leakage, winsorization, quintile monotonicity
│
└── config/
    └── settings.py                 # Centralised paths + model hyperparameters
```

---

## EVQTI v3 — EV Quant Transition Index

### The intuition

Norway ranks low on EVQTI despite being a global EV leader. Countries like Brazil, Indonesia and South Africa rank high. The reason: early-stage markets grow faster — Bass diffusion law applied to the country cross-section. EVQTI operationalises this into a measurable, anti-leakage, statistically validated factor.

### Factor construction

Signal built at t-1. Target: EV growth at t+1. No future information leaks in.

| Feature (t-1) | Direction | Weight | Empirical IC |
|---|---|---|---|
| EV stock share | Negative — lower = earlier stage | **48.5%** | −0.259 (p<0.001) |
| EV sales share | Negative — lower = earlier stage | **51.5%** | −0.275 (p<0.001) |

Growth rate excluded: IC = −0.037, p = 0.36 — no cross-country predictive power.

### Normalisation pipeline

1. Winsorise features at 1%–99%
2. Create lags (t-1) on winsorised values
3. Re-winsorise lags at their own P1/P99
4. Z-score cross-section per year
5. Score = −(w₁·z₁ + w₂·z₂)
6. Winsorise target at 5%–95%

### Validation

| Metric | Result | Threshold |
|---|---|---|
| Rank IC | +0.237 | > 0.10 = robust |
| IR | 0.99 | > 0.50 = consistent |
| % years IC > 0 | 93% | — |
| Q5-Q1 spread | +122pp | — |
| Rolling backtest | +0.16 → +0.52 | Strengthening |

---

## ETII — EV Transition Intelligence Index

Combines EVQTI (momentum, 50%) with World Bank Fundamentals Score (structural capacity, 50%) into a single 0–100 composite. Weight is adjustable via slider.

**Fundamentals Score** = equally-weighted composite of GDP per capita, urbanisation, GDP growth and CO2 per capita from the World Bank public API (no key required, cached 24h).

**Resilience:** if the World Bank API is unavailable, ETII falls back to EVQTI-only ranking with a neutral fundamentals score of 50 — the page always loads, never crashes.

**Opportunity matrix:** scatter of EVQTI vs Fundamentals, with quadrant annotations. Top-right (high both) = highest-conviction early-stage opportunities.

---

## Infrastructure Intelligence

Powered by IEA charging point data (52 countries, 82.3M total chargers in 2025).

| Metric | Description |
|---|---|
| Total chargers | Publicly available, all types combined |
| EVs per charger | IEA pre-computed ratio — lower = better coverage |
| Charger mix | Fast / Slow / Ultra-fast breakdown |
| Infrastructure race | Indexed growth: chargers vs EV sales — who is keeping pace? |

**IEA reference target:** 10 EVs per public charger (shown as reference line).

---

## Aggregation rules — no double-counting

```
IEA 2026: powertrain='EV' = aggregate (BEV + PHEV + FCEV)
          powertrain='BEV'/'PHEV'... = components

WRONG:  Total = EV + BEV + PHEV   ← double-counts
RIGHT:  Total = EV                ← aggregate only
   or:  Total = BEV + PHEV + FCEV ← components only

IEA 2025: no 'EV' aggregate in EV sales → BEV + PHEV + FCEV
IEA charging: mode='EVSE' (not 'EV') — corrected in aggregator

Primary key: country × category × year × parameter × mode × powertrain
```

---

## Fuel price context

IEA Energy Prices Monthly Excerpt (Jun 2026) — end-user prices including taxes.

| Coverage | 33 countries · Jan 2015–May 2026 · USD/litre |
|---|---|
| Products | Gasoline · Diesel |
| Exclusions | Ukraine (insufficient data coverage) |

---

## EV Pulse setup

```toml
# .streamlit/secrets.toml
NEWSAPI_KEY      = "your_key_here"       # newsapi.org — 100 req/day free
ANTHROPIC_API_KEY = "your_key_here"      # optional — enables AI summaries
```

On Streamlit Cloud: App Settings → Secrets → paste contents.

**Resilience:** news cached 1h, AI summaries cached 24h. Every failure mode (timeout, invalid key, rate limit) surfaces a specific message — never a blank page.

---

## Dataset comparison

| Mode | Countries | Coverage | EV sales schema |
|---|---|---|---|
| IEA 2026 | 63 | 2010–2025 | powertrain = `EV` (aggregate) |
| IEA 2025 | 54 | 2010–2024 | powertrain = `BEV / PHEV / FCEV` |
| Combined | 63 | 2010–2025 | 2026 priority on overlaps |

**Automatic fallback:** if selected dataset fails to load, app automatically tries the other year and notifies the user — a partial dataset is always preferred over a blank page.

---

## Test suite

29 regression tests covering the most critical invariants:

- **Double-counting** — Norway 2023 Cars: EV aggregate (110,000) must equal Aggregator output
- **Schema detection** — `powertrain='EV'` in charging points must NOT trigger 2026 schema
- **Anti-leakage** — lag at year t must equal value at year t-1, not t
- **Winsorization** — lag values must stay within their own P1/P99 distribution
- **Quintile monotonicity** — Q1 growth < Q2 < Q3 < Q4 < Q5 (factor validity)

Run with: `pytest tests/`

---

## Local setup

```bash
git clone https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-
cd EV-Transition-Intelligence-Lab-ETIL-
pip install -r requirements.txt
streamlit run app.py
```

---

## Dependencies

```
streamlit>=1.28.0    pandas>=2.2.0       numpy>=1.26.0
scipy>=1.11.0        scikit-learn>=1.3.0  plotly>=5.18.0
statsmodels>=0.14.0  openpyxl>=3.1.0     requests>=2.31.0
pytest>=7.4.0
```

---

## License

MIT — see [LICENSE](LICENSE).  
IEA data © OECD/IEA. Non-commercial research only.  
EVQTI and ETII are research constructs, not financial advice.

---

**Carlos Eduardo** · Quant Data Scientist · Edumetria  
[LinkedIn](https://www.linkedin.com/in/carlos-eduardo-2753a92b5) · [GitHub](https://github.com/edu-moraess) · [App](https://quantev.streamlit.app)

*IEA GEVO 2025/2026 · IEA Energy Prices 2026 · World Bank API · Models: GMM · GP · Bass · Monte Carlo · EVQTI · ETII*
