# ⚡ EV Transition Intelligence Lab (ETIL)

Plataforma quantitativa institucional para análise da transição global para veículos elétricos.

**Fonte de dados:** IEA Global EV Outlook 2025 e 2026  
**Deploy:** [quantev.streamlit.app](https://quantev.streamlit.app)  
**Autor:** [Carlos Eduardo](https://www.linkedin.com/in/carlos-eduardo-2753a92b5) · [GitHub](https://github.com/edu-moraess)

---

## Visão geral

O ETIL combina modelos de difusão tecnológica, detecção de regimes e um fator quantitativo proprietário — o **EVQTI** — para mapear o estágio de transição EV de 60+ países com rigor estatístico institucional.

O projeto parte de uma premissa econômica clara: **o crescimento futuro está nos mercados que ainda não chegaram lá, não nos que já chegaram.** O EVQTI operacionaliza isso em um fator mensurável, anti-leakage e validado empiricamente.

---

## Resultados em produção (IEA 2026, Cars)

| Métrica | Valor |
|---|---|
| Rank IC médio | **+0.140** |
| % ICs positivos | 71% |
| % ICs significativos (p<0.05) | 21% |
| Spread Q5-Q1 | **+50.0pp** |
| Monotônico por quintil | ✅ Q1=62.5% → Q5=112.5% |

O IC se torna estável e crescente a partir de 2021 — quando o mercado EV atingiu massa crítica para o sinal funcionar consistentemente.

---

## Estrutura do projeto

```
├── app.py                        # Entrada principal — roteamento e pipeline
├── requirements.txt
├── .streamlit/
│   └── config.toml               # Tema nativo Streamlit
│
├── data/
│   └── raw/
│       ├── EV_Data_Explorer_2025_GEVO_clean.csv
│       └── EV_Data_Explorer_2026_GEVO_clean.csv
│
├── ingestion/
│   ├── dataset_engine.py         # Multi-dataset: 2025, 2026, combinado
│   └── loader.py
│
├── preprocessing/
│   ├── cleaning.py
│   ├── transformation.py
│   └── features.py
│
├── models/
│   ├── diffusion/                # Bass, Logistic, Gompertz
│   ├── forecasting/              # Gaussian Process + Monte Carlo
│   ├── regimes/                  # GMM + Bayesian Changepoint
│   └── scoring/
│       └── evqti.py              # EVQTI v2 — fator quant anti-leakage
│
├── analytics/
│   ├── country_analysis.py
│   ├── global_ranking.py
│   └── scenario_engine.py
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

## Páginas

| Página | Descrição |
|---|---|
| 🌍 Global Monitor | Mapa coroplético, KPIs globais, evolução temporal, top 10 países |
| 🔬 Country Lab | Série histórica, YoY, EV stock share, modelos de difusão, regimes |
| 🧠 Diffusion Engine | Ajuste Bass / Logistic / Gompertz com projeções configuráveis |
| 🔄 Regime Detection | GMM com probabilidades por regime + Bayesian Changepoint |
| 📊 Forecast Lab | Gaussian Process + Monte Carlo com cenários e percentis |
| 🏆 EVQTI Ranking | IC, Rank IC, análise de quintis e ranking completo por país |

---

## EVQTI — EV Quant Transition Index v2

Fator quantitativo anti-leakage construído sobre variáveis defasadas (t-1).

**Lógica econômica:** países em estágio early-stage de difusão (baixa penetração + momentum positivo) concentram o maior crescimento futuro — lei de difusão de Bass aplicada ao cross-section de países.

**Construção:**

| Feature (t-1) | Direção | Peso | Justificativa |
|---|---|---|---|
| ev_stock_share | Negativo | 45% | Baixa penetração = mais espaço de crescimento |
| ev_sales_share | Negativo | 35% | Complementar ao stock share |
| growth_rate | Positivo | 20% | Momentum recente como confirmação |

**Validação empírica:**
- Quintis perfeitamente monotônicos: Q1=62.5% → Q5=112.5% de crescimento futuro médio
- IC se estabiliza e cresce pós-2021, quando o mercado EV ganhou massa crítica
- ICs negativos em 2011-2013 são esperados: universo pequeno demais para sinal cross-country

---

## Multi-dataset

| Modo | Países | Cobertura |
|---|---|---|
| IEA 2026 | 63 | 2010-2025 |
| IEA 2025 | 54 | 2010-2024 |
| Combinado | 63 | 2010-2025 (2026 com prioridade) |

Schema detectado automaticamente: IEA 2025 usa `BEV/PHEV/FCEV`, IEA 2026 usa `EV` agregado.

---

## Setup local

```bash
git clone https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-
cd EV-Transition-Intelligence-Lab-ETIL-
pip install -r requirements.txt
streamlit run app.py
```

---

## Dependências

```
streamlit>=1.28.0 · pandas>=2.2.0 · numpy>=1.26.0
scipy>=1.11.0 · scikit-learn>=1.3.0 · plotly>=5.18.0
statsmodels>=0.14.0 · openpyxl>=3.1.0
```

> `hmmlearn` removido por incompatibilidade com Python 3.14. Detecção de regimes via `GaussianMixture` (scikit-learn) com interface equivalente.

---

## Autor

**Carlos Eduardo**  
Quant Data Scientist · Edumetria

[LinkedIn](https://www.linkedin.com/in/carlos-eduardo-2753a92b5) · [GitHub](https://github.com/edu-moraess) · [App](https://quantev.streamlit.app)

---

*Fonte: IEA Global EV Outlook 2025/2026 · Modelos: GMM · GP · Bass · Monte Carlo · EVQTI v2*
