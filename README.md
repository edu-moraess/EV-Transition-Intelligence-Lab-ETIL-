EV Transition Intelligence Lab (ETIL)
Plataforma institucional quantitativa avançada para análise global da transição para veículos elétricos
Aplicação em Produção: quantev.streamlit.app
Autor: Carlos Eduardo · GitHub
Fontes de Dados: IEA Global EV Outlook 2025/2026 · IEA Energy Prices 2026 · World Bank API · NewsAPI
Visão Geral
O ETIL constitui uma plataforma analítica institucional de última geração projetada para monitorar, quantificar e prever a transição global para veículos elétricos em 63 países.
A arquitetura integra modelos de difusão estocástica (Bass, Logistic, Gompertz), detecção bayesiana de regimes, Gaussian Processes com simulação Monte Carlo, análise multidimensional de infraestrutura e um sistema proprietário de índices compostos (EVQTI + ETII) em um ambiente unificado, resiliente e escalável.
Desenvolvida com rigor científico extremo, a plataforma oferece inteligência prospectiva sobre dinâmica de adoção, estágios de maturidade e trajetórias futuras da transição energética, mantendo total imparcialidade analítica e reprodutibilidade estatística.
Funcionalidades Principais
Página
Objetivo Analítico
🌍 Global Monitor
Visão macro da adoção, velocidades de crescimento e composição da frota global
🔬 Country Lab
Deep-dive por país: vendas, momentum, contexto energético, fundamentos World Bank, modelos de difusão e regime de adoção
🧠 Diffusion Engine
Posicionamento na curva-S com Bass, Logistic e Gompertz; métricas de ajuste (R², AIC, BIC) e comparação normalizada multi-país
🔄 Regime Detection
Classificação dinâmica de mercados (Emerging, Expanding, Accelerating, Maturing) via GMM + Bayesian Changepoint
📊 Forecast Lab
Previsões probabilísticas de 10 anos com IC 80%/95% (Gaussian Process + Monte Carlo, 2.000 trajetórias)
🏆 EVQTI Ranking
Ranking preditivo de mercados early-stage com validação completa de Information Coefficient (IC), Information Ratio (IR) e backtest rolling
📰 EV Pulse
Digest de notícias em tempo real com resumos analíticos gerados por IA (requer NewsAPI)
🔌 Infrastructure
Inteligência de rede de carregamento: total de chargers, EVs/charger vs. meta IEA, mix tecnológico e corrida infraestrutura × vendas
🎯 Scenarios
Simulações “What-if” com parâmetros ajustáveis do modelo Bass (p, q, M) e bandas de incerteza
🌐 ETII Composite Index
Índice 0–100 que combina momentum (EVQTI) e capacidade estrutural (World Bank) com matriz de oportunidades
🔬 Research Intelligence
Rastreamento acadêmico de papers, citações e tendências na literatura de transição energética
Resultados de Produção — EVQTI v3 (IEA 2026, Cars)
Métrica
Valor
Interpretação
Mean Rank IC
+0.237
Sinal preditivo robusto (threshold > 0.10)
Information Ratio
0.99
Consistência temporal elevada (target > 0.5)
Anos com IC positivo
93%
13 de 14 anos com sinal direcionalmente correto
Q5-Q1 growth spread
+122 pp
Quintil superior cresceu 122 pontos percentuais acima do inferior
Monotonicidade de quintis
✅
Q1=49% → Q2=87% → Q3=106% → Q4=122% → Q5=171%
Rolling backtest IC
+0.16 → +0.52
Fortalecimento progressivo do sinal
EVQTI v3 — EV Quant Transition Index
O EVQTI operacionaliza a lei de difusão de Bass aplicada transversalmente entre países: mercados em estágios iniciais exibem maior potencial de crescimento relativo. Países líderes consolidados (ex.: Noruega) apresentam scores moderados, enquanto economias emergentes com alta capacidade (Brasil, Indonésia, África do Sul etc.) destacam-se.
Construção do Fator (t-1 → crescimento t+1, sem leakage):
Feature (t-1)
Direção
Peso
IC Empírico
EV stock share
Negativa
48,5%
-0.259 (p<0.001)
EV sales share
Negativa
51,5%
-0.275 (p<0.001)
Pipeline de Normalização (anti-leakage):
Winsorização 1%-99%
Lag dos valores winsorizados
Re-winsorização dos lags
Z-score cross-sectional anual
Score composto com inversão de sinal
Winsorização final do target (5%-95%)
ETII — EV Transition Intelligence Index
Índice composto 0–100 que integra:
50% EVQTI (momentum dinâmico)
50% World Bank Fundamentals Score (capacidade estrutural: PIB per capita, taxa de urbanização, crescimento do PIB e intensidade de CO₂)
Matriz de Oportunidades: scatter plot EVQTI × Fundamentals com quadrantes anotados. O quadrante superior-direito identifica as oportunidades de maior convicção na transição.
Resiliência: fallback automático para EVQTI puro (fundamentals neutro = 50) em caso de indisponibilidade da API World Bank.
Inteligência de Infraestrutura
Baseada em dados IEA de pontos de carregamento (52 países, 82,3 milhões de chargers em 2025).
Métricas chave:
Total de carregadores públicos
EVs por charger (meta IEA de referência: 10 EVs/charger)
Mix por tecnologia (Fast / Slow / Ultra-fast)
Índice de corrida infraestrutura × vendas de EVs
Regras rigorosas de agregação (zero double-counting):
IEA 2026: powertrain='EV' = agregado correto (BEV + PHEV + FCEV)
Componentes utilizados apenas quando granularidade é requerida
Contexto de Preços de Combustíveis
IEA Energy Prices Monthly (Jun/2026) — preços finais ao consumidor (com impostos) para gasolina e diesel em 33 países (Jan/2015–Mai/2026).
Arquitetura Técnica
Projeto modular, testável e resiliente com ênfase em reprodutibilidade e validação estatística rigorosa:
ingestion/ — carregamento e agregação dimensional robusta
preprocessing/ — feature engineering avançado (CAGR, momentum, volatilidade, penetração)
models/ — Bass, Logistic, Gompertz, Gaussian Process, GMM, Bayesian Changepoint, Monte Carlo
analytics/ — engines de cenário e ranking
tests/ — 29 testes de regressão críticos (anti-leakage, monotonicidade, double-counting)
Setup Local
git clone https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-
cd EV-Transition-Intelligence-Lab-ETIL-
pip install -r requirements.txt
streamlit run app.py
Licença: MIT
Dados IEA: © OECD/IEA — uso exclusivo para pesquisa não-comercial.
EVQTI e ETII: construtos analíticos de pesquisa. Não constituem recomendação financeira ou de investimento.
Carlos Eduardo
Quant Data Scientist | Edumetria
LinkedIn · GitHub · Aplicação