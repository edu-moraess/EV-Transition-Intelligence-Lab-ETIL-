# EVQTI — Relatório de validação de pesquisa (P1.7–P1.10)

**Projeto:** EV Transition Intelligence Lab (ETIL)  
**Componente:** EV Quantitative Transition Index (EVQTI)  
**Período da avaliação:** 2026-11-26 a 2026-11-27  
**Painel:** IEA Global EV Data Explorer 2026 — Historical, Cars  
**Branch de pesquisa avaliada:** research/etil-evqti-sensitivity-robustness  
**Base conhecida da branch:** 5223a9cea9be9880febcdc0c040fed000343e940  
**Natureza:** relatório de pesquisa exploratória; não altera pesos nem declara validação preditiva definitiva.

> **Nota de escopo para a main:** este documento registra resultados obtidos na branch de pesquisa indicada. Sua inclusão na main versiona o relatório, mas não promove automaticamente para a main as alterações de implementação existentes na branch de pesquisa. A comparação entre as branches mostrou 32 commits de diferença à época da consolidação; portanto, nenhuma dessas alterações de código foi mesclada por meio deste relatório.

---

## 1. Resumo executivo

Entre P1.7 e P1.10, o EVQTI apresentou associação positiva entre seu score e o crescimento futuro de vendas de EV no cross-section de países. O sinal permaneceu positivo:

- em nove origens temporais consecutivas (2016–2024) no rolling-origin;
- em seis configurações de pesos avaliadas na análise de sensibilidade;
- nas 2.000 reamostragens por cenário do country-cluster bootstrap P1.9;
- após a exclusão individual de cada um dos 60 países;
- no painel balanceado de 48 países presentes em todos os nove anos; e
- nas 2.000 reamostragens do painel balanceado P1.10.

Esses resultados são evidência favorável de **robustez associativa dentro do painel disponível**. Não demonstram causalidade, não garantem previsões futuras e não constituem validação plenamente point-in-time, pois não há histórico completo de datas de publicação e revisões dos releases IEA.

**Decisão de pesquisa:** manter os pesos de referência atuais (estoque 0,485; vendas 0,515). Não alterar os pesos de produção com base apenas na vantagem marginal do cenário sales-only.

## 2. Dados e protocolo

A ingestão IEA 2026 foi validada com 49.161 linhas brutas, 72 países e categorias Historical / Projection-CPS / Projection-STEPS. A análise do EVQTI utilizou o painel Historical, Cars, com 60 países e anos de 2010 a 2025 (760 linhas no painel informado para P1.7/P1.8).

O EVQTI usa duas features defasadas:

- ev_stock_share_lag1;
- ev_sales_share_lag1.

A direção econômica do score é invertida: menor penetração corrente recebe score maior, sob a hipótese de que mercados em estágio inicial de difusão podem apresentar maior crescimento futuro. O target de avaliação é growth_t1, crescimento de vendas no ano seguinte. O target bruto é usado na avaliação principal; growth_t1_win é diagnóstico de sensibilidade.

Os pesos de referência são 0,485 para share de estoque e 0,515 para share de vendas. São pesos descritivos derivados de ICs no painel histórico completo IEA 2026 Cars (2010–2025), e **não** pesos point-in-time ou estimativas independentes out-of-sample.

### 2.1 Rolling-origin (P1.7)

Para cada origem de teste T, os pesos são calibrados apenas com labels de anos anteriores observáveis até T. Os ICs direcionais anuais de treino determinam os pesos normalizados; se não houver sinal positivo, o procedimento prevê fallback equal-weight explícito. O score na origem T é avaliado contra crescimento bruto em T+1.

A avaliação cobre nove origens, 2016–2024. Essa disciplina reduz leakage de seleção de pesos, mas não resolve revisões históricas do IEA.

## 3. Resultados

### 3.1 P1.7 — rolling-origin temporal

Rank IC anual por ano de teste:

| Ano | Rank IC |
|---:|---:|
| 2016 | 0,2400 |
| 2017 | 0,1880 |
| 2018 | 0,1872 |
| 2019 | 0,0989 |
| 2020 | 0,1940 |
| 2021 | 0,4783 |
| 2022 | 0,4801 |
| 2023 | 0,5748 |
| 2024 | 0,3764 |

Resumo informado: Rank IC médio 0,3131; mediano 0,2400; 9/9 anos positivos. O spread médio top-minus-bottom informado foi 338,67 (unidade do target de crescimento percentual, conforme a saída de pesquisa). Esse spread é sensível a valores extremos e não deve ser interpretado isoladamente como tamanho de efeito típico.

### 3.2 P1.8 — sensibilidade de pesos e target

Rank IC médio por configuração (target bruto; resultados praticamente iguais com target winsorizado):

| Configuração | Rank IC médio | Anos positivos |
|---|---:|---:|
| Pesos iguais (50/50) | 0,3157 | 9/9 |
| Referência oficial (48,5/51,5) | 0,3165 | 9/9 |
| 75% vendas / 25% estoque | 0,3242 | 9/9 |
| Apenas vendas | 0,3291 | 9/9 |
| 75% estoque / 25% vendas | 0,3066 | 9/9 |
| Apenas estoque | 0,2797 | 8/9 |

O target bruto e o winsorizado produziram Rank ICs muito próximos. Em contrapartida, a winsorização reduziu substancialmente a magnitude do spread médio, indicando influência de valores extremos sobre essa métrica de magnitude, sem alteração equivalente da ordenação por ranks.

A estabilidade média de ranking entre anos (correlação de ranks) variou conforme os pesos: stock-only foi mais estável (aprox. 0,956), enquanto sales-only foi menos estável (aprox. 0,909). Os pesos oficiais e iguais tiveram estabilidade praticamente idêntica (aprox. 0,945).

**Interpretação:** sales-only apresentou o maior Rank IC médio, mas menor estabilidade de ranking. Stock-only foi mais estável, porém com associação média menor. A configuração oficial representa um compromisso razoável entre as duas dimensões; a diferença frente a equal-weight é mínima.

### 3.3 P1.9 — incerteza por reamostragem e influência dos países

Foi executado country-cluster bootstrap com 2.000 replicações por cenário, reamostrando países e mantendo juntas as observações temporais de cada cluster.

| Cenário | Rank IC médio observado | Bootstrap 95% |
|---|---:|---:|
| Referência oficial | 0,3165 | [0,2242; 0,3878] |
| Pesos iguais | 0,3157 | [0,2244; 0,3911] |
| Apenas vendas | 0,3291 | [0,2322; 0,4105] |
| Apenas estoque | 0,2797 | [0,1946; 0,3613] |

Todos os intervalos percentis de 95% ficaram acima de zero. A média de Rank IC foi positiva em todas as 2.000 replicações para cada cenário. Essa fração descreve as replicações bootstrap observadas; **não** deve ser comunicada como probabilidade posterior de sucesso ou garantia de previsão.

Leave-one-country-out foi executado para os 60 países. Nenhuma remoção individual tornou negativo o Rank IC médio. A mediana da variação absoluta do Rank IC médio foi 0,0022 (equal-weight), 0,0024 (oficial), 0,0031 (sales-only) e 0,0038 (stock-only). A maior variação absoluta observada foi 0,0285 (stock-only). Rússia, Indonésia, Noruega e África do Sul apareceram entre as observações mais influentes em diferentes configurações. Influência não implica erro ou inadequação desses países; apenas indica sensibilidade da estatística à composição cross-sectional.

Cobertura: 48 países estavam presentes em todos os nove anos; 52 estavam presentes em pelo menos sete anos; a mediana foi nove anos.

### 3.4 P1.10 — painel balanceado

A análise foi repetida com os mesmos 48 países elegíveis em todas as nove origens temporais.

| Cenário | Rank IC médio — completo | Rank IC médio — balanceado | Diferença |
|---|---:|---:|---:|
| Pesos iguais | 0,3157 | 0,3137 | -0,0020 |
| Referência oficial | 0,3165 | 0,3149 | -0,0016 |
| Apenas vendas | 0,3291 | 0,3253 | -0,0038 |
| Apenas estoque | 0,2797 | 0,2899 | +0,0102 |

Todos os cenários tiveram Rank IC positivo em 9/9 anos no painel balanceado. O menor Rank IC anual foi 0,1249 para pesos oficiais, 0,1197 para pesos iguais, 0,0567 para sales-only e 0,0268 para stock-only.

Country-cluster bootstrap balanceado (2.000 replicações por cenário):

| Cenário | Países | Rank IC médio | Bootstrap 95% |
|---|---:|---:|---:|
| Referência oficial | 48 | 0,3149 | [0,2178; 0,3939] |
| Pesos iguais | 48 | 0,3137 | [0,2149; 0,4033] |
| Apenas vendas | 48 | 0,3253 | [0,2329; 0,4053] |
| Apenas estoque | 48 | 0,2899 | [0,1871; 0,3730] |

Os intervalos permaneceram acima de zero. As diferenças pequenas entre painel completo e balanceado sugerem que a associação observada não é explicada apenas pela entrada e saída de países na amostra.

## 4. Decisões metodológicas

1. **Manter pesos oficiais:** 0,485 estoque / 0,515 vendas.
2. Manter sales-only, stock-only e pesos iguais como cenários de sensibilidade, não como substitutos automáticos do índice de referência.
3. Usar growth_t1 bruto como target primário da avaliação; manter target winsorizado como diagnóstico de sensibilidade.
4. Preservar a distinção entre ranking atual com pesos fixos (diagnóstico descritivo in-sample) e backtest rolling-origin com calibração train-only.
5. Descrever resultados como associação positiva e robustez exploratória, não como causalidade ou previsão comprovada.
6. Não modificar os pesos nem o pipeline de produção com base exclusivamente nestes resultados.

## 5. Limitações e riscos remanescentes

- **Vintages IEA:** anos de observação não são datas de publicação. Releases posteriores podem revisar valores históricos. Sem snapshots point-in-time e datas de disponibilidade, o backtest não é fully vintage-aware.
- **Número de janelas:** são nove origens anuais; as janelas são temporalmente dependentes e não representam nove experimentos independentes.
- **Inferência cross-sectional:** bootstrap por país quantifica sensibilidade à reamostragem de países no painel observado, mas não elimina dependência regional/econômica, choques comuns ou heterogeneidade estrutural.
- **Múltiplas comparações:** p-values anuais e de cenários são nominais; não houve correção por múltiplos testes. Não devem ser usados como evidência confirmatória isolada.
- **Seleção de especificação:** as configurações foram examinadas após exploração do painel. Os resultados não substituem uma especificação pré-registrada e uma validação externa.
- **Generalização:** a evidência cobre países e anos presentes no IEA 2026 Cars; não estabelece desempenho para outros segmentos, regiões, releases ou períodos futuros.
- **Spread:** a média top-bottom é sensível a caudas e winsorização; deve ser acompanhada por mediana, distribuição e incerteza.
- **Bootstrap:** intervalos percentis são condicionais à amostra e ao esquema de reamostragem adotado; não incorporam incerteza de revisão do dataset, escolha de modelo ou seleção de features.

## 6. Próximas etapas recomendadas

1. Obter e estruturar vintages históricos IEA com data de publicação e revisões, se disponíveis.
2. Congelar uma especificação de pesquisa antes de uma nova avaliação externa.
3. Avaliar estabilidade em outro release temporal, segmento ou fonte independente, mantendo a separação entre treino e teste.
4. Documentar dependência regional e conduzir análises de sensibilidade por grupos econômicos/geográficos, sem remover países apenas por influência.
5. Só então considerar uma revisão formal dos pesos ou qualquer afirmação mais forte sobre desempenho preditivo.

## 7. Reprodutibilidade e artefatos

As saídas P1.8, P1.9 e P1.10 foram geradas em diretórios temporários do Colab (/tmp/ETIL_EVQTI_P18_20261126T234137Z, /tmp/ETIL_EVQTI_P19_20261126T235634Z e /tmp/ETIL_EVQTI_P110_20261127T001235Z). Esses diretórios são ambientes temporários e **não estão incluídos neste commit**. Os resultados tabulados neste documento foram transcritos das saídas fornecidas durante a execução.

Parâmetros principais:
- P1.9: seed 20261126; 2.000 replicações; 60 clusters de país.
- P1.10: seed 20261127; 2.000 replicações; 48 clusters comuns.
- Período de avaliação: 2016–2024.
- Target primário: growth_t1.

Nenhum dado sintético foi introduzido nesta avaliação. Este relatório não afirma que os artefatos temporários foram arquivados nem substitui a necessidade de preservar CSVs/JSONs de execução em armazenamento versionado caso a rastreabilidade integral seja exigida.
