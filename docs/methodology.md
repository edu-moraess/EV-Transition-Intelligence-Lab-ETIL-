# Metodologia

## Modelos de difusão

O ETIL ajusta curvas Bass, Logistic e Gompertz a séries cumulativas de vendas de veículos elétricos. As métricas AIC, BIC, RMSE, MAPE e R² apresentadas pela aplicação são métricas de ajuste dentro da amostra; não constituem validação preditiva fora da amostra. O teto de mercado é estimado pela curva e não deve ser interpretado como uma capacidade econômica observada sem hipóteses externas.

## Detecção de regimes

A classe `HMMRegimeDetector` usa um **Gaussian Mixture Model (GMM)** com componentes ordenados pela primeira feature de adoção. Ela não estima uma cadeia de Markov nem probabilidades de transição latentes de um HMM. A matriz de transição exibida é uma frequência empírica de estados adjacentes na sequência observada.

A classe `BayesianChangePoint` é mantida por compatibilidade de API, mas implementa uma triagem heurística de deslocamentos padronizados de média. Seus scores normalizados não são probabilidades posteriores e não identificam causalidade econômica.

## Previsão e cenários

O Forecast Lab combina Gaussian Process Regression, amostragem Monte Carlo e cenários condicionados a regimes. Os caminhos podem ser correlacionados pela covariância posterior do GP e reproduzidos com uma semente local. No motor condicionado a regimes, os intervalos são **quantis empíricos de simulação** que dependem do modelo, dos priors heurísticos de regime e dos parâmetros de cenário. Eles não são automaticamente intervalos de confiança ou de credibilidade.

O Scenario Engine aplica choques explícitos aos parâmetros `p`, `q` e `M` do modelo Bass. Os cenários são sensibilidades do modelo, não previsões causais nem projeções de política pública.

## EVQTI e ETII

O EVQTI é um fator de pesquisa baseado em shares de estoque e vendas defasados em um ano, com target de crescimento no ano seguinte. As features e o target passam por winsorização e o score usa normalização cross-section e pesos calibrados por Information Coefficient. O resultado é um sinal associativo; não é garantia de crescimento, recomendação de investimento ou inferência causal.

O ETII combina o EVQTI normalizado com um score de fundamentos obtido de indicadores do Banco Mundial quando existe cobertura. Os fundamentos são um contexto comparativo, com normalização por faixas definidas no código. O ano dos indicadores é limitado ao ano de referência do painel EV quando essa informação é fornecida. Falhas ou cobertura parcial da API devem permanecer visíveis na interpretação do índice.

## Dados e missingness

As fontes principais são os releases locais do IEA Global EV Data Explorer. Dados ausentes são preservados e não devem ser interpretados automaticamente como zero. O modo `Total` agrega todos os modos de veículo e calcula shares ponderados pelos denominadores reconstruídos a partir das vendas/estoque EV e dos shares publicados; a cobertura dessas shares é exposta no painel para indicar quando algum modo não possui denominador válido.
