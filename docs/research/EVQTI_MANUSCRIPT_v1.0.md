# EVQTI: desenvolvimento e avaliação temporal de um índice quantitativo para análise da transição global para veículos elétricos

**Versão:** 1.0  
**Data:** 27 de novembro de 2026  
**Projeto:** EV Transition Intelligence Lab (ETIL)  
**Natureza:** pesquisa independente aplicada; manuscrito não submetido a revisão por pares.

## Resumo

Este estudo documenta o desenvolvimento e a avaliação temporal do Electric Vehicle Transition Quality Index (EVQTI), um indicador quantitativo desenvolvido no âmbito do EV Transition Intelligence Lab. Na implementação examinada, o índice combina participações defasadas de veículos elétricos (EVs) no estoque e nas vendas para ordenar países segundo seu potencial relativo de crescimento subsequente das vendas. A análise principal utiliza dados históricos de automóveis (Cars) do Global EV Data Explorer da International Energy Agency (IEA), release 2026. O painel processado contém 60 países e 760 observações entre 2010 e 2025. Um backtest rolling-origin com seis anos de calibração e um ano de teste produz nove janelas anuais, de 2016 a 2024. O Rank Information Coefficient (Rank IC) médio é 0,3131, positivo nas nove janelas; quatro janelas apresentam p-value nominal inferior a 0,05. Em análise complementar sobre uma coorte comum de 45 países, o EVQTI é comparado com persistência do crescimento e indicadores simples de baixa penetração. Um bootstrap pareado por clusters de países, com 2.000 réplicas e recalibração integral dos pesos em cada réplica, estima Rank IC médio de 0,3416 para o EVQTI. A diferença média em relação à persistência é 0,2059 (IC bootstrap de 95%: 0,0940 a 0,3120). As diferenças em relação aos baselines de baixa penetração de vendas (-0,0132; IC: -0,0386 a 0,0147) e de estoque (+0,0104; IC: -0,0158 a 0,0393) incluem zero. Os resultados indicam associação direcional consistente e vantagem em relação ao baseline de persistência, mas não demonstram ganho incremental sobre os indicadores simples de penetração. O estudo é exploratório, não causal e não plenamente vintage-aware, pois não utiliza vintages históricos de publicação da IEA.

**Palavras-chave:** veículos elétricos; difusão tecnológica; indicador composto; avaliação temporal; Rank IC; bootstrap por clusters; IEA.

## 1. Introdução

A transição para veículos elétricos apresenta trajetórias heterogêneas entre países. Diferenças no estágio de adoção e na estrutura dos mercados tornam relevante investigar como medidas atuais de penetração se relacionam com o crescimento subsequente.

Modelos de difusão tecnológica descrevem a evolução da adoção ao longo do tempo, enquanto indicadores compostos sintetizam múltiplas dimensões em uma medida comparativa. Entretanto, a construção de um índice não garante validade empírica: escolhas de normalização, ponderação, cobertura e tratamento temporal podem afetar rankings e interpretações.

O EVQTI foi desenvolvido para investigar uma hipótese específica: países com menor participação de EVs no estoque e nas vendas podem apresentar maior crescimento proporcional subsequente, compatível com uma dinâmica de catch-up. O índice, na implementação avaliada, não mede todas as dimensões da qualidade da transição energética. Seu significado operacional é mais restrito: uma ordenação relativa de potencial de crescimento futuro das vendas, condicionada às features e à amostra utilizadas.

Este estudo documenta a especificação examinada, a avaliação temporal, a comparação com baselines simples e a incerteza estimada por reamostragem de países.

## 2. Objetivos e hipótese

O objetivo principal é avaliar se o score EVQTI apresenta associação monotônica positiva com o crescimento subsequente das vendas de EVs em avaliações temporais sucessivas.

A hipótese direcional é que menores participações defasadas de EVs no estoque e nas vendas estejam associadas a maior crescimento futuro das vendas. Trata-se de hipótese associativa; não implica que baixa penetração cause crescimento.

Objetivos complementares:
- avaliar a consistência do sinal em janelas anuais rolling-origin;
- comparar o índice com persistência do crescimento e indicadores univariados de penetração;
- quantificar incerteza por bootstrap pareado de clusters de países, recalibrando o índice em cada réplica;
- explicitar limitações de cobertura, dependência, revisões históricas e generalização.

## 3. Dados

A fonte é o Global EV Data Explorer da International Energy Agency, release 2026. A análise principal utiliza a categoria **Historical** e o modo **Cars**, com entidades agregadas excluídas. O painel processado contém 60 países, 760 observações e anos entre 2010 e 2025.

As variáveis centrais são:
- participação de EVs no estoque (`ev_stock_share`);
- participação de EVs nas vendas (`ev_sales_share`);
- crescimento anual das vendas (`growth_rate`).

Os lags são construídos por país e ano. Ausências não são interpretadas como zero nem preenchidas por interpolação temporal. A taxa de crescimento é calculada a partir das vendas anuais consecutivas, conforme a implementação de referência.

A IEA publica releases revisados. Embora o release 2026 seja a fonte da análise principal, o conjunto utilizado não contém snapshots históricos de cada vintage de publicação nem timestamps de disponibilidade por observação. Portanto, a avaliação respeita a ordem temporal das variáveis e dos labels no painel disponível, mas não reproduz integralmente o conjunto de informações que estaria disponível em tempo real em cada origem histórica.

## 4. Construção do EVQTI

### 4.1 Features e orientação

O score utiliza duas features:
- `ev_stock_share_lag1`;
- `ev_sales_share_lag1`.

Ambas recebem direção negativa: valores menores de penetração contribuem para um score maior, interpretado como maior potencial relativo de catch-up. O crescimento futuro das vendas (`growth_t1`) é o alvo de avaliação, não uma feature do score.

### 4.2 Pré-processamento temporal

Para cada ano de referência, as features defasadas são winsorizadas nos quantis de 1% e 99%, utilizando somente observações históricas anteriores àquele ano. Quando há menos de dez observações históricas disponíveis para estimar os limites, a winsorização não é aplicada.

O alvo futuro bruto é preservado para avaliação. Uma cópia winsorizada do alvo, com limites de 5% e 95% estimados apenas a partir de labels anteriores à origem, é mantida como variável diagnóstica; ela não substitui o alvo bruto no backtest descrito neste manuscrito.

### 4.3 Calibração e score

Na origem de teste T, a calibração considera os seis anos anteriores, T-6 a T-1. Para cada feature, calcula-se em cada ano de treino a correlação de Spearman entre a feature defasada e o crescimento subsequente. O sinal é orientado conforme a direção econômica definida para a feature. A média dos ICs anuais válidos forma o sinal direcional de treino.

Somente contribuições direcionais positivas são utilizadas para formar os pesos, normalizados para somar um. Se nenhum sinal direcional positivo estiver disponível, aplica-se fallback de pesos iguais entre as features calibráveis. A calibração utiliza apenas labels cujo resultado subsequente é observável até a origem de teste.

No ano de teste, cada feature é imputada pela mediana cross-sectional daquele ano, quando necessário, e padronizada por z-score dentro da própria seção transversal. O score é a combinação ponderada dos z-scores orientados. A normalização utiliza apenas os valores das features no ano de teste, não os labels futuros.

### 4.4 Avaliação rolling-origin

A avaliação usa `train_years=6`, `test_years=1`, `step=1` e `min_countries=10`. O alvo é `growth_t1` bruto. O Rank IC é a correlação de Spearman entre score e alvo no cross-section de cada ano. Também são calculados correlação de Pearson, p-value nominal de Spearman e spread entre os quintis extremos.

As nove janelas de teste abrangem 2016 a 2024. Como cada janela contém um único ano, o desvio-padrão temporal dentro de cada janela é zero por construção e não deve ser interpretado como estimativa de volatilidade interanual.

## 5. Comparação com baselines

A análise comparativa utiliza uma coorte comum de 45 países com observações completas nas nove janelas de teste (2016–2024), totalizando 369 observações país-ano. São considerados quatro sinais:
1. EVQTI;
2. persistência, definida pelo crescimento observado no ano T (`growth_rate`) para ordenar o crescimento no ano T+1;
3. baixa penetração de vendas, definida pelo negativo da participação defasada nas vendas;
4. baixa penetração de estoque, definida pelo negativo da participação defasada no estoque.

As comparações são cross-sectional por ano e resumidas pela média dos nove Rank ICs anuais. O resultado desta análise é distinto da avaliação principal de 60 países: aqui, a calibração do EVQTI também é restrita à coorte comum de 45 países, para manter coerência entre calibração e avaliação.

## 6. Bootstrap pareado por clusters

Foi executado bootstrap pareado por país, com 2.000 réplicas e seed 20261127. Em cada réplica, 45 países são reamostrados com reposição. A mesma amostra de clusters é mantida ao longo dos anos, preservando a estrutura longitudinal dentro de cada país. O pipeline do EVQTI é recalibrado em cada réplica usando somente os anos de treino disponíveis antes de cada origem; a normalização cross-sectional e os scores são recalculados. Os baselines são avaliados sobre a mesma amostra reamostrada.

Os intervalos apresentados são intervalos percentis empíricos de 95%. As probabilidades bootstrap de uma diferença ser positiva são descritivas e não devem ser confundidas com p-values ou com uma decisão formal de superioridade. A inferência permanece condicionada ao painel histórico final disponível e não incorpora incerteza de vintages da IEA.

## 7. Resultados

### 7.1 Avaliação temporal principal — painel de 60 países

O backtest produziu nove janelas anuais, 2016–2024, todas com Rank IC positivo. A média foi **0,31308**. Quatro das nove janelas apresentaram p-value nominal inferior a 0,05. Os p-values não foram ajustados para multiplicidade.

| Ano de teste | Rank IC | p-value nominal | n |
|---:|---:|---:|---:|
| 2016 | 0,2400 | 0,096651 | 49 |
| 2017 | 0,1880 | 0,182028 | 52 |
| 2018 | 0,1872 | 0,183994 | 52 |
| 2019 | 0,0989 | 0,472494 | 55 |
| 2020 | 0,1940 | 0,155779 | 55 |
| 2021 | 0,4783 | 0,000146 | 58 |
| 2022 | 0,4801 | 0,000119 | 59 |
| 2023 | 0,5748 | 0,000002 | 59 |
| 2024 | 0,3764 | 0,003299 | 59 |

O sinal foi positivo em todas as janelas, mas sua magnitude variou. A evidência estatística nominal concentrou-se nos anos mais recentes do período avaliado.

### 7.2 Baselines — coorte comum de 45 países

A média observada dos Rank ICs anuais foi:

| Sinal | Rank IC médio |
|---|---:|
| EVQTI | 0,34159 |
| Persistência | 0,13565 |
| Baixa penetração de vendas | 0,35484 |
| Baixa penetração de estoque | 0,33118 |

O EVQTI superou o baseline de persistência em magnitude média. O baseline univariado de baixa penetração de vendas apresentou média ligeiramente superior ao EVQTI. A diferença entre os indicadores simples e o EVQTI é pequena e deve ser interpretada em conjunto com a incerteza pareada.

### 7.3 Incerteza bootstrap dos modelos

| Modelo | Rank IC médio observado | IC bootstrap 95% |
|---|---:|---:|
| EVQTI | 0,3416 | [0,2303; 0,4369] |
| Persistência | 0,1356 | [0,0197; 0,2458] |
| Baixa penetração de vendas | 0,3548 | [0,2464; 0,4504] |
| Baixa penetração de estoque | 0,3312 | [0,2163; 0,4289] |

### 7.4 Diferenças pareadas

| Comparação | Diferença média | IC bootstrap 95% | P(diferença > 0) |
|---|---:|---:|---:|
| EVQTI − persistência | +0,2059 | [+0,0940; +0,3120] | 0,9995 |
| EVQTI − baixa penetração de vendas | −0,0132 | [−0,0386; +0,0147] | 0,1880 |
| EVQTI − baixa penetração de estoque | +0,0104 | [−0,0158; +0,0393] | 0,8155 |

A diferença em relação à persistência permaneceu positiva no intervalo bootstrap observado. As diferenças contra os indicadores simples de baixa penetração incluem zero. Portanto, os resultados não demonstram vantagem incremental do EVQTI sobre esses baselines.

## 8. Discussão

Os resultados sustentam uma associação monotônica positiva entre o score EVQTI e o crescimento futuro das vendas no desenho avaliado. A positividade em nove janelas consecutivas é evidência de consistência direcional no período, embora não signifique que cada janela tenha associação estatisticamente significativa.

A comparação com baselines é essencial para interpretar o valor incremental do índice. O EVQTI apresenta desempenho superior ao baseline de persistência do crescimento, mas não demonstra superioridade em relação ao sinal simples de baixa penetração nas vendas. Isso sugere que parte importante da informação capturada pelo índice pode estar relacionada ao estágio atual de adoção. A combinação de estoque e vendas pode oferecer uma representação mais estruturada, mas os resultados disponíveis não comprovam que ela acrescente poder associativo significativo em relação ao indicador univariado mais forte.

A análise bootstrap com recalibração integral é mais abrangente que uma reamostragem que mantém os pesos fixos: ela incorpora variação na amostra de países e na calibração dos pesos, condicionada ao conjunto de dados utilizado. Ainda assim, os nove anos e os mesmos países ao longo do tempo produzem dependências entre janelas que devem ser consideradas na interpretação. O intervalo percentil e a probabilidade bootstrap são evidências exploratórias, não uma prova formal de superioridade generalizável.

## 9. Limitações

1. **Vintages da IEA:** não há snapshots históricos e timestamps de publicação por observação para reconstruir exatamente a informação disponível em cada origem.
2. **Dependência temporal:** janelas anuais consecutivas e observações repetidas dos mesmos países não são independentes.
3. **Tamanho da amostra:** a comparação com baselines usa 45 clusters de países e nove anos-alvo.
4. **Multiplicidade:** p-values anuais são nominais e não corrigidos para múltiplas comparações.
5. **Cobertura:** a disponibilidade varia por país e ano; a coorte completa de 45 países é uma restrição da comparação.
6. **Associação, não causalidade:** Rank IC não identifica mecanismos causais nem efeitos de políticas.
7. **Generalização:** os resultados de 2016–2024 não garantem desempenho em períodos futuros ou releases diferentes.
8. **Definição do índice:** na versão avaliada, EVQTI mede potencial relativo de crescimento/catch-up baseado em penetração, não qualidade multidimensional abrangente da transição energética.
9. **Bootstrap:** os intervalos são condicionais ao painel histórico final e não capturam incerteza de revisões ou vintages de dados.
10. **Comparabilidade de resultados:** o Rank IC de 0,31308 no painel de 60 países e o de 0,34159 na coorte de 45 países pertencem a desenhos distintos e não devem ser comparados como se fossem a mesma amostra.

## 10. Conclusão

O EVQTI apresentou associação positiva entre seu score e o crescimento futuro das vendas de veículos elétricos em nove janelas rolling-origin de 2016 a 2024. Na comparação com baselines sobre uma coorte comum de 45 países, o índice demonstrou vantagem em relação à persistência do crescimento, mas não demonstrou ganho incremental em relação aos indicadores simples de baixa penetração de vendas ou estoque.

A contribuição principal deste estágio é a documentação de uma especificação operacional, de uma avaliação temporal e de uma comparação transparente com alternativas simples, incluindo recalibração em bootstrap pareado por países. Os resultados justificam continuidade da investigação, mas não sustentam alegações de causalidade, superioridade geral ou validação preditiva definitiva.

## Referências

- Bass, F. M. (1969). A new product growth model for consumer durables. *Management Science, 15*(5), 215–227. https://doi.org/10.1287/mnsc.15.5.215
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *The Review of Economics and Statistics, 90*(3), 414–427. https://doi.org/10.1162/rest.90.3.414
- International Energy Agency. (2026). *Global EV Data Explorer*. https://www.iea.org/data-and-statistics/data-tools/global-ev-data-explorer
- Nardo, M., Saisana, M., Saltelli, A., Tarantola, S., Hoffmann, A., & Giovannini, E. (2008). *Handbook on Constructing Composite Indicators: Methodology and User Guide*. OECD Publishing / European Commission Joint Research Centre. https://doi.org/10.1787/9789264043466-en
- Rogers, E. M. (2003). *Diffusion of Innovations* (5th ed.). Free Press.
- Spearman, C. (1904). The proof and measurement of association between two things. *The American Journal of Psychology, 15*(1), 72–101. https://doi.org/10.2307/1412159
- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting, 16*(4), 437–450. https://doi.org/10.1016/S0169-2070(00)00065-0

## Reprodutibilidade e estado do código

A implementação de referência está associada ao commit `75e43af68f87d449bf83436976e856144059b4e6` da branch `main` do repositório ETIL. O backtest usa `EVQTI.backtest` com `train_years=6`, `test_years=1`, `step=1` e `min_countries=10`.

Os resultados consolidados e os parâmetros estão registrados em [EVQTI_EXPERIMENT_RECORD_2026-11-27.md](EVQTI_EXPERIMENT_RECORD_2026-11-27.md), e o procedimento de reprodução está descrito em [EVQTI_REPRODUCIBILITY.md](EVQTI_REPRODUCIBILITY.md). Os nove artefatos experimentais originais em CSV/JSON foram arquivados, junto ao manifesto SHA-256 e à nota de reprodução, no pacote [EVQTI_RESEARCH_ARTIFACTS_2026-11-27.zip](artifacts/EVQTI_RESEARCH_ARTIFACTS_2026-11-27.zip). O ZIP foi validado e confirmado nesta branch; os arquivos individuais permanecem compactados, sem serem duplicados como arquivos separados no repositório.

**Nota de nomenclatura:** este manuscrito preserva a denominação por extenso adotada na versão editorial fechada: *Electric Vehicle Transition Quality Index*. A docstring atual de `models/scoring/evqti.py` ainda contém a denominação anterior “EV Quant Transition Index”; a harmonização do nome no código e nos demais materiais deve ser feita em alteração editorial explícita, não implicitamente neste registro.
