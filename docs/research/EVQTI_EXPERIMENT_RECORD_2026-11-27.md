# EVQTI — registro experimental consolidado

**Data:** 27/11/2026  
**Código de referência:** 75e43af68f87d449bf83436976e856144059b4e6  
**Branch:** main  
**Fonte:** IEA Global EV Data Explorer 2026 — Historical, Cars.

## 1. Avaliação rolling-origin principal

| Parâmetro | Valor |
|---|---:|
| Países no painel processado | 60 |
| Observações | 760 |
| Período | 2010–2025 |
| Treino | 6 anos |
| Teste | 1 ano |
| Step | 1 ano |
| Mínimo de países por janela | 10 |
| Janelas de teste | 2016–2024 (9) |
| Rank IC médio | 0,3130799499 |
| Janelas com Rank IC positivo | 9/9 |
| Janelas com p nominal < 0,05 | 4/9 |

### Resultados por ano

| Ano | Rank IC (ic_mean) | p-value | n |
|---:|---:|---:|---:|
| 2016 | 0.240043 | 0.096651 | 49 |
| 2017 | 0.187982 | 0.182028 | 52 |
| 2018 | 0.187154 | 0.183994 | 52 |
| 2019 | 0.098907 | 0.472494 | 55 |
| 2020 | 0.194026 | 0.155779 | 55 |
| 2021 | 0.478283 | 0.000146 | 58 |
| 2022 | 0.480127 | 0.000119 | 59 |
| 2023 | 0.574773 | 0.000002 | 59 |
| 2024 | 0.376425 | 0.003299 | 59 |

A coluna ic_mean do retorno anual de backtest contém a correlação de Spearman entre score e alvo; neste desenho anual ela é o Rank IC. Como cada janela contém um único ano, ic_std=0 por construção.

## 2. Comparação com baselines — coorte comum

A comparação corrigida utiliza persistência contemporânea (growth_rate no ano T) para ordenar crescimento em T+1. O painel comum contém 45 países e 369 observações país-ano entre 2016 e 2024. No bootstrap de pipeline completo, o EVQTI é recalibrado sobre a mesma coorte, razão pela qual o resultado difere ligeiramente da análise V2 que calibrava sobre o painel mais amplo.

| Sinal | Rank IC médio observado |
|---|---:|
| EVQTI | 0.34159 |
| Persistência (growth_rate em T) | 0.13565 |
| Baixa penetração de vendas | 0.35484 |
| Baixa penetração de estoque | 0.33118 |

## 3. Bootstrap pareado por país — pipeline completo

- Réplicas: 2.000; válidas: 2.000/2.000.
- Clusters: 45 países; anos: 2016–2024.
- Seed: 20261127.
- Tempo reportado: 410,8 segundos.
- A mesma amostra de países é usada para os quatro sinais e ao longo dos anos.
- O pipeline EVQTI é recalibrado em cada réplica usando somente anos de treino anteriores à origem; scores e normalização cross-sectional são recalculados.
- Intervalos: percentis empíricos de 2,5% e 97,5%.

### Intervalos por modelo

| Sinal | Média observada | IC bootstrap 95% |
|---|---:|---:|
| EVQTI | 0.3416 | [0.2303; 0.4369] |
| Persistência | 0.1356 | [0.0197; 0.2458] |
| Baixa penetração de vendas | 0.3548 | [0.2464; 0.4504] |
| Baixa penetração de estoque | 0.3312 | [0.2163; 0.4289] |

### Diferenças pareadas

| Diferença | Média | IC bootstrap 95% | Proporção bootstrap positiva |
|---|---:|---:|---:|
| EVQTI − persistência | +0.2059 | [+0.0940; +0.3120] | 0.9995 |
| EVQTI − baixa penetração de vendas | −0.0132 | [−0.0386; +0.0147] | 0.1880 |
| EVQTI − baixa penetração de estoque | +0.0104 | [−0.0158; +0.0393] | 0.8155 |

A proporção bootstrap positiva é descritiva e não é um p-value formal.

## 4. Pesos recalibrados por origem

| Ano | Peso stock | Peso sales |
|---:|---:|---:|
| 2016 | 0.6029 | 0.3971 |
| 2017 | 0.5024 | 0.4976 |
| 2018 | 0.5561 | 0.4439 |
| 2019 | 0.5225 | 0.4775 |
| 2020 | 0.4671 | 0.5329 |
| 2021 | 0.4760 | 0.5240 |
| 2022 | 0.4470 | 0.5530 |
| 2023 | 0.4676 | 0.5324 |
| 2024 | 0.4668 | 0.5332 |

## 5. Interpretação autorizada

1. O EVQTI apresentou Rank IC positivo nas nove janelas anuais do painel de 60 países.
2. Na coorte comum de 45 países, o EVQTI superou o baseline de persistência.
3. Não foi demonstrada superioridade incremental sobre os indicadores simples de baixa penetração de vendas ou estoque: os intervalos pareados incluem zero.
4. Os resultados são associativos e condicionais à amostra e à especificação.
5. A análise não é plenamente vintage-aware: não utiliza snapshots históricos de publicação da IEA.
6. Os resultados de 60 países (média 0,31308) e da coorte de 45 países (média 0,34159) pertencem a desenhos distintos.

## 6. Artefatos gerados no Colab ainda não arquivados

- /content/EVQTI_WALK_FORWARD_MAIN_75e43af.csv
- /content/EVQTI_WALK_FORWARD_SUMMARY_75e43af.json
- /content/ETIL_EVQTI_MAIN_WALK_FORWARD.csv
- /content/ETIL_EVQTI_MAIN_SUMMARY.json
- /content/ETIL_EVQTI_BASELINE_COMPARISON_V2.csv
- /content/ETIL_EVQTI_BASELINE_SUMMARY_V2.json
- /content/ETIL_EVQTI_FULL_PIPELINE_BOOTSTRAP.csv
- /content/ETIL_EVQTI_FULL_PIPELINE_MODEL_INTERVALS.csv
- /content/ETIL_EVQTI_FULL_PIPELINE_BOOTSTRAP.json

Esses arquivos não estão incorporados neste commit. A inclusão dos originais é necessária para arquivar as réplicas e tabelas completas; não foram reconstruídos CSVs a partir das estatísticas resumidas. Nenhum dado bruto da IEA foi copiado para este registro.
