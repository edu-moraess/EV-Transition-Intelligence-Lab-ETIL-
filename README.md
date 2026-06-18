# ⚡ EV Transition Intelligence Lab (ETIL)

Plataforma quantitativa institucional para análise da transição global para veículos elétricos.

**Fonte de dados:** IEA Global EV Outlook 2025 e 2026  
**Deploy:** [quantev.streamlit.app](https://quantev.streamlit.app)  
**Autor:** [Carlos Eduardo](https://www.linkedin.com/in/carlos-eduardo-2753a92b5) · [GitHub](https://github.com/edu-moraess)

---

## Visão geral

O ETIL combina modelos de difusão tecnológica, detecção de regimes e um fator quantitativo proprietário — o **EVQTI** — para mapear o estágio de transição EV de 60+ países com rigor estatístico institucional.

A premissa central: **o crescimento futuro concentra-se nos mercados que ainda não chegaram lá, não nos que já chegaram.** O EVQTI operacionaliza essa lei de difusão em um fator mensurável, anti-leakage e validado empiricamente.

---

## Resultados em produção — EVQTI v3 (IEA 2026, Cars)

| Métrica | Valor | Interpretação |
|---|---|---|
| Rank IC médio | **+0.237** | Sinal robusto — limiar de utilidade é 0.10 |
| Information Ratio (IR) | **0.99** | IR > 0.5 = consistente ao longo do tempo |
| % anos IC positivo | **93%** | Sinal falha em menos de 1 de cada 14 anos |
| Spread Q5-Q1 | **+122pp** | Diferença de crescimento futuro entre quintis extremos |
| Quintis monotônicos | ✅ | Q1=49% → Q2=87% → Q3=106% → Q4=122% → Q5=171% |

---

## Estrutura do projeto

```
├── app.py                        # Entrada principal — roteamento e pipeline
├── requirements.txt
├── .streamlit/
│   └── config.toml               # Tema light institucional — navy #1f3a6e
│
├── data/
│   └── raw/
│       ├── EV_Data_Explorer_2025_GEVO_clean.csv
│       └── EV_Data_Explorer_2026_GEVO_clean.csv
│
├── ingestion/
│   ├── dataset_engine.py         # Multi-dataset 2025/2026/combinado
│   └── loader.py                 # Carregamento com fallback automático
│
├── preprocessing/
│   ├── cleaning.py
│   ├── transformation.py
│   └── features.py               # Feature engineering: growth, CAGR, momentum
│
├── models/
│   ├── diffusion/                # Bass, Logistic, Gompertz (com R²)
│   ├── forecasting/              # Gaussian Process + Monte Carlo (IC 80/95%)
│   ├── regimes/                  # GMM (proxy HMM) + Bayesian Changepoint
│   └── scoring/
│       └── evqti.py              # EVQTI v3 — pesos calibrados por IC empírico
│
├── analytics/
│   ├── country_analysis.py
│   ├── global_ranking.py
│   └── scenario_engine.py
│
├── visualization/
│   ├── theme.py                  # Paleta quant institucional centralizada
│   └── __init__.py               # Lazy import
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
| 🌍 Global Monitor | Mapa EV stock share, KPIs globais, evolução temporal, top 10 países |
| 🔬 Country Lab | Série histórica, YoY, stock share, difusão com R², regimes |
| 🧠 Diffusion Engine | Bass / Logistic / Gompertz com R², projeções e comparativo multi-país |
| 🔄 Regime Detection | GMM com probabilidades por regime + Bayesian Changepoint |
| 📊 Forecast Lab | Gaussian Process + MC com IC 80%/95%, percentis por ano e cenários |
| 🏆 EVQTI Ranking | IC, Rank IC, t-stat, IR, quintis com erro padrão e ranking completo |

---

## EVQTI v3 — EV Quant Transition Index

### Lógica econômica

Países em estágio early-stage de difusão EV — baixa penetração atual com demanda latente — concentram o maior crescimento futuro. É a lei de Bass aplicada ao cross-section de países: o mercado saturado cresce menos; o mercado emergente cresce mais.

### Construção do fator (anti-leakage)

Sinal calculado com features defasadas (t-1). Target é o crescimento em t+1.

| Feature (t-1) | Direção | Peso | IC empírico |
|---|---|---|---|
| ev_stock_share | Negativo (early-stage) | **48.5%** | −0.259 (p<0.001) |
| ev_sales_share | Negativo (early-stage) | **51.5%** | −0.275 (p<0.001) |

> `growth_rate_lag1` foi excluída: IC = −0.037, p = 0.36 — não significativo. Momentum de 1 ano é ruído no cross-section de países.

### Pipeline de normalização

1. Winsorização 1%–99% nas features raw
2. Lags (t-1) criados sobre os valores winsorizados
3. Lags re-winsorizados com P1/P99 da distribuição do próprio lag
4. Z-score cross-section por ano
5. Score = −(w₁·z₁ + w₂·z₂) — sinal negativo converte early-stage em positivo
6. Target winsorizado em 5%–95%

### Validação estatística

- **IC (Pearson sobre ranks)** e **Rank IC (Spearman)** calculados por ano
- **t-stat** por ano: t = IC · √(n−2) / √(1−IC²)
- **Information Ratio** = IC_médio / σ(IC)
- **Análise de quintis** com barra de erro (erro padrão)
- **Spread Q5-Q1** como medida de separação do fator

### Interpretação do IC

| Faixa | Significado |
|---|---|
| \|IC\| < 0.05 | Sem sinal útil |
| \|IC\| 0.05–0.10 | Sinal fraco |
| \|IC\| > 0.10 | Sinal robusto ✓ |
| IR > 0.5 | Consistente ao longo do tempo ✓ |

**EVQTI v3: IC = +0.237 · IR = 0.99**

---

## Multi-dataset

| Modo | Países | Cobertura | Schema IEA |
|---|---|---|---|
| IEA 2026 | 63 | 2010–2025 | powertrain = `EV` (agregado) |
| IEA 2025 | 54 | 2010–2024 | powertrain = `BEV / PHEV / FCEV` |
| Combinado | 63 | 2010–2025 | 2026 com prioridade em sobreposições |

Schema detectado automaticamente via presença de `EV` em `EV sales`.

---

## Design system

A paleta é definida centralmente em `visualization/theme.py` e importada por todas as páginas — nenhum hex hardcoded nas páginas individuais.

| Elemento | Cor | Uso |
|---|---|---|
| Primary / positivo | `#1f3a6e` (navy) | Barras, linhas principais, ranking |
| Negativo / alerta | `#8b1a1a` (vermelho escuro) | IC negativo, crescimento negativo |
| Momentum | `#b7770d` (âmbar) | Changepoints, alertas moderados |
| Positivo suave | `#1a4731` (verde escuro) | Crescimento, regimes de aceleração |
| Texto secundário | `#4a4a4a` (ardósia) | Labels, anotações |

Tema Streamlit: light mode com `primaryColor = #1f3a6e`.

---

## Setup local

```bash
git clone https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-
cd EV-Transition-Intelligence-Lab-ETIL-
pip install -r requirements.txt
streamlit run app.py
```

Os arquivos de dados devem estar em `data/raw/`.

---

## Dependências

```
streamlit>=1.28.0   pandas>=2.2.0      numpy>=1.26.0
scipy>=1.11.0       scikit-learn>=1.3.0 plotly>=5.18.0
statsmodels>=0.14.0 openpyxl>=3.1.0
```

> `hmmlearn` removido — incompatível com Python 3.14. Detecção de regimes via `GaussianMixture` (scikit-learn), interface equivalente.

---

## Autor

**Carlos Eduardo**  
Quant Data Scientist · Edumetria

[LinkedIn](https://www.linkedin.com/in/carlos-eduardo-2753a92b5) · [GitHub](https://github.com/edu-moraess) · [App](https://quantev.streamlit.app)

---

*Fonte: IEA Global EV Outlook 2025/2026 · Modelos: GMM · GP · Bass · Monte Carlo · EVQTI*
