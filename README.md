**EV Transition Intelligence Lab (ETIL)**
  
*Plataforma institucional quantitativa para análise global da transição para veículos elétricos*

**Aplicação em produção:** [quantev.streamlit.app](https://quantev.streamlit.app)  
**Autor:** Carlos Eduardo · [GitHub](https://github.com/edu-moraess)

**Fontes de dados:** IEA Global EV Outlook 2025/2026 · IEA Energy Prices 2026 · World Bank API · NewsAPI

---

### Visão Geral

O **ETIL** constitui uma plataforma analítica avançada que monitora a transição global para veículos elétricos em **63 países**. A solução integra modelos de difusão, detecção de regimes, análise de infraestrutura, simulação de políticas e um índice composto proprietário (EVQTI + ETII) em um ambiente unificado de pesquisa.

Projetada para rigor analítico extremo e acessibilidade, a plataforma permite que decisores, pesquisadores e investidores compreendam não apenas *onde* a adoção está ocorrendo, mas *com que velocidade* e *para onde* o sistema está evoluindo.

---

### Funcionalidades Principais

| Página | Objetivo Analítico |
|--------|--------------------|
| 🌍 **Global Monitor** | Visão macro da adoção, ranking de crescimento e composição da frota |
| 🔬 **Country Lab** | Análise aprofundada por país: vendas, momentum, contexto de combustíveis, fundamentos World Bank, modelos de difusão e regime de adoção |
| 🧠 **Diffusion Engine** | Posicionamento de cada país na curva-S (Bass, Logistic, Gompertz) com métricas de ajuste (R², AIC, BIC) e comparação normalizada multi-país |
| 🔄 **Regime Detection** | Classificação de mercados (Emerging, Expanding, Accelerating, Maturing) via GMM + Bayesian Changepoint |
| 📊 **Forecast Lab** | Previsões probabilísticas de 10 anos com intervalos de confiança 80%/95% (Gaussian Process + Monte Carlo, 2.000 trajetórias) |
| 🏆 **EVQTI Ranking** | Ranking preditivo de mercados early-stage com validação IC/IR completa e backtest rolling |
| 📰 **EV Pulse** | Digest de notícias em tempo real com resumos analíticos gerados por IA (requer chave NewsAPI) |
| 🔌 **Infrastructure** | Inteligência de rede de carregamento: total de chargers, EVs por charger vs. meta IEA, mix por tipo e corrida infraestrutura × vendas |
| 🎯 **Scenarios** | Simulações "What if" via parâmetros ajustáveis do modelo Bass (p, q, M) com bandas de incerteza |
| 🌐 **ETII Composite Index** | Índice 0–100 que combina momentum (EVQTI) e capacidade estrutural (World Bank) com matriz de oportunidades |
| 🔬 **Research Intelligence** | Rastreamento acadêmico de papers, citações e tendências na literatura de transição energética |

---

### Resultados de Produção — EVQTI v3 (IEA 2026, Cars)

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| Mean Rank IC | +0.237 | Sinal preditivo robusto (threshold > 0.10) |
| Information Ratio | 0.99 | Consistência temporal elevada (target > 0.5) |
| Anos com IC positivo | 93% | 13 de 14 anos com sinal direcionalmente correto |
| Q5-Q1 growth spread | +122 pp | Quintil superior cresceu 122 pontos percentuais acima do inferior |
| Monotonicidade de quintis | ✅ | Q1=49% → Q2=87% → Q3=106% → Q4=122% → Q5=171% |
| Rolling backtest IC | +0.16 → +0.52 | Fortalecimento do sinal ao longo do tempo |

---

### EVQTI v3 — EV Quant Transition Index

**Princípio:** Países em estágios iniciais apresentam maior potencial de crescimento relativo (lei de difusão de Bass aplicada transversalmente). A Noruega, apesar de líder absoluta, ranqueia moderadamente no EVQTI, enquanto Brasil, Indonésia e África do Sul aparecem entre os mais promissores.

**Construção do Fator (sem leakage de informação futura):**
- **EV stock share** (t-1): peso 48,5% | IC = -0.259 (p<0.001)
- **EV sales share** (t-1): peso 51,5% | IC = -0.275 (p<0.001)

**Pipeline de normalização:**
1. Winsorização 1%-99%
2. Lag dos valores winsorizados
3. Re-winsorização dos lags
4. Z-score cross-sectional anual
5. Score composto com inversão de sinal
6. Winsorização final do target (5%-95%)

---

### ETII — EV Transition Intelligence Index

Índice composto 0–100 que pondera:
- **50% EVQTI** (momentum)
- **50% World Bank Fundamentals Score** (capacidade estrutural: PIB per capita, urbanização, crescimento do PIB e CO₂ per capita)

**Matriz de Oportunidades:** gráfico de dispersão EVQTI × Fundamentals com quadrantes claramente anotados. Quadrante superior-direito identifica as oportunidades de maior convicção.

**Resiliência:** em caso de indisponibilidade da API World Bank, o sistema degrada graciosamente para EVQTI puro (score fundamental neutro = 50).

---

### Inteligência de Infraestrutura

Baseada em dados IEA de pontos de carregamento (52 países, 82,3 milhões de chargers em 2025).

**Métricas principais:**
- Total de carregadores públicos
- EVs por charger (meta IEA de referência: 10 EVs/charger)
- Mix por tecnologia (Fast / Slow / Ultra-fast)
- Índice de corrida: crescimento de chargers × crescimento de vendas

**Regras rigorosas de agregação (sem double-counting):**
- IEA 2026: `powertrain='EV'` representa o agregado correto (BEV + PHEV + FCEV).
- Componentes são utilizados apenas quando desejada granularidade.

---

### Contexto de Preços de Combustíveis

IEA Energy Prices Monthly (Jun/2026) — preços ao consumidor final (incluindo impostos) para gasolina e diesel em 33 países (Jan/2015–Mai/2026).

---

### Arquitetura Técnica

O projeto segue arquitetura modular e testável, com forte ênfase em reprodutibilidade, resiliência e validação estatística.

**Principais módulos:**
- `ingestion/` — carregamento e agregação dimensional robusta
- `preprocessing/` — feature engineering avançado (CAGR, momentum, volatilidade, penetração)
- `models/` — Bass, Logistic, Gompertz, Gaussian Process, GMM, Bayesian Changepoint, Monte Carlo
- `analytics/` — engines de cenário e ranking
- `tests/` — 29 testes de regressão críticos (anti-leakage, monotonicidade, double-counting)

---

### Setup Local

```bash
git clone https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-
cd EV-Transition-Intelligence-Lab-ETIL-
pip install -r requirements.txt
streamlit run app.py
```

**Dependências principais:** Streamlit, Pandas, NumPy, SciPy, scikit-learn, Plotly, Statsmodels, etc.

---

**Licença:** MIT  
**Dados IEA:** © OECD/IEA — uso exclusivo para pesquisa não-comercial.  
**EVQTI e ETII:** construtos analíticos de pesquisa. Não constituem recomendação de investimento.

---

**Carlos Eduardo**  
Quant Data Scientist | Edumetria  
[LinkedIn](https://www.linkedin.com/in/carlos-eduardo-2753a92b5) · [GitHub](https://github.com/edu-moraess) · [Aplicação](https://quantev.streamlit.app)
