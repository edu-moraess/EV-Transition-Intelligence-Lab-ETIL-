# Relatório Técnico de Auditoria — ETIL

**Data da auditoria:** 13 de agosto de 2026  
**Escopo:** repositório `EV-Transition-Intelligence-Lab-ETIL-`, branch `main`  
**Commit auditado inicialmente:** `0b20bf10259f31f2b32a9ec0d0749a62a44622e5`

## Adendo de remediação — 17 de agosto de 2026

A branch `fix/technical-audit-remediation` implementou as correções prioritárias da auditoria. O modo `Total` agora é servido pelo `IEALoader` canônico, calcula shares ponderados pelos denominadores implícitos e expõe cobertura parcial. O `FactorEngine` limita winsorização e calibração ao cutoff temporal configurável; o World Bank pode consultar um ano de referência alinhado ao painel EV; o fallback do app está restrito a falhas recuperáveis; a proteção XSRF foi reativada no devcontainer; a documentação metodológica foi sincronizada; `requests` foi declarado diretamente; e os nomes de arquivos com espaço final foram normalizados.

A suíte atual contém **44 testes aprovados**. A validação de compilação, pipeline real para Cars/2-and-3-wheelers/Total e integração analítica foi executada após as alterações. Backtests walk-forward completos para cada origem temporal e seleção formal de número de regimes continuam sendo próximos passos metodológicos, não foram tratados como resolvidos por este patch.

## Resumo executivo

O ETIL é uma aplicação Streamlit com uma base de código materialmente mais ampla do que a cobertura de testes sugeria. A auditoria mapeou **69 arquivos Python** e aproximadamente **9.077 linhas Python**, sem ciclos de importação locais detectados. A suíte inicial continha 31 testes e passava após a instalação das dependências declaradas; ao final da auditoria original, a suíte passou a conter **37 testes**. Após o adendo de remediação, a suíte contém **44 testes**, todos aprovados.

A maior fragilidade encontrada não era estética: coexistiam duas rotas de ingestão com semânticas diferentes. A aplicação principal utilizava `DatasetEngine`, enquanto páginas específicas utilizavam `IEALoader` e `Aggregator`. Essa divergência alterava a agregação de shares e eliminava os carregadores do dataset IEA 2025 por assumir indevidamente o rótulo de modo `EVSE`. A refatoração transformou `DatasetEngine` em uma fachada compatível sobre o carregador canônico e corrigiu a agregação por modo/powertrain.

Os modelos de difusão, previsão e regime eram funcionais, mas apresentavam riscos metodológicos importantes. Foram adicionadas validações de domínio ao Bass, trajetórias GP correlacionadas e reprodutíveis, simulação de regimes sem incrementos artificiais nulos no primeiro ano e terminologia estatisticamente honesta para quantis de simulação e pontos de mudança. Ainda assim, o projeto não deve apresentar esses módulos como evidência causal, nem como forecasting plenamente validado fora da amostra.

> **Conclusão:** o repositório saiu de uma condição tecnicamente promissora, porém com divergências de dados e alegações metodológicas excessivas, para uma base mais coerente, rastreável e defensável. A prioridade seguinte deve ser a validação temporal formal e o endurecimento do índice multifatorial contra possíveis vieses retrospectivos.

## Arquitetura e fluxo de execução

O fluxo efetivo é `app.py` → `DatasetEngine` → `IEALoader`/`Aggregator` → `DataCleaner` → `DataTransformer` → `FeatureEngineer` → páginas Streamlit. As camadas de analytics, modelos de difusão, forecast, regimes e scoring são consumidas diretamente por páginas específicas. O fluxo agora é coerente com a arquitetura alvo abaixo, embora algumas páginas ainda chamem fontes auxiliares diretamente para combustível e indicadores macroeconômicos.

| Camada | Componentes principais | Avaliação após a auditoria |
|---|---|---|
| Aplicação | `app.py`, `streamlit_pages/` | A inicialização headless foi concluída com sucesso. A lógica quantitativa continua parcialmente distribuída pelas páginas. |
| Dados | `ingestion/loader.py`, `ingestion/aggregator.py`, `ingestion/dataset_engine.py` | Consolidada em torno de `IEALoader` e `Aggregator`; `DatasetEngine` agora preserva compatibilidade sem duplicar regras. |
| Pré-processamento | `cleaning.py`, `transformation.py`, `features.py` | Imputação silenciosa removida. Features são ordenadas por país-ano e denominadores de infraestrutura foram corrigidos. |
| Difusão | Bass, Logistic, Gompertz | Bass ganhou validação, restrições e diagnóstico básico; comparação ainda é predominantemente in-sample. |
| Forecast | GP, Monte Carlo, regime forecast | Simulações usam covariância do GP e RNG local; quantis são descritos corretamente como empíricos. |
| Regimes | GMM e detector de mudanças | GMM é explicitamente tratado como GMM. A tela não chama mais uma heurística de mean shift de “Bayesian”. |
| Scoring | EVQTI e `FactorEngine` | Compatibilidade do output EVQTI foi corrigida; robustez temporal do fator multifatorial permanece pendente. |

## Auditoria de dados

Os arquivos IEA locais não apresentaram chaves primárias duplicadas ou valores nulos no campo bruto `value`. O dataset IEA 2025 contém 16.424 linhas, 63 países/regiões e cobertura de 2010–2030; o IEA 2026 contém 49.161 linhas, 72 países/regiões e cobertura de 2010–2035. As unidades observadas nas métricas centrais são `Vehicles` para vendas/estoque e `percent` para shares. O Global EV Data Explorer declara que cobre vendas, estoque e infraestrutura de carregamento, com histórico e projeções, o que é consistente com essas dimensões. [1]

| Validação do painel histórico | IEA 2025 Cars | IEA 2026 Cars | Combinado Cars |
|---|---:|---:|---:|
| Linhas após agregação | 652 | 760 | 760 |
| Países | 53 | 60 | 60 |
| Intervalo temporal | 2010–2024 | 2010–2025 | 2010–2025 |
| Duplicidades em `country × year` | 0 | 0 | 0 |
| Shares fora de 0–100 | 0 | 0 | 0 |
| Linhas sem `charging_points` | 269 | 323 | 323 |

A cobertura de infraestrutura é estruturalmente incompleta, portanto valores ausentes foram preservados. O ETIL não mais interpola `ev_sales` nem preenche automaticamente carregadores, população, PIB ou outras variáveis auxiliares com medianas de país. Essa decisão evita transformar ausência de observação em sinal quantitativo artificial.

### Achados e correções de engenharia de dados

| Severidade | Achado | Evidência e impacto | Correção aplicada |
|---|---|---|---|
| **HIGH** | Duas rotas de dados com agregações diferentes | A aplicação usava `DatasetEngine`, enquanto outras páginas usavam `IEALoader`; shares e carregadores podiam divergir entre telas. | `DatasetEngine` tornou-se fachada sobre o pipeline canônico. |
| **HIGH** | Carregadores IEA 2025 excluídos | O agregador exigia `mode='EVSE'`, mas o IEA 2025 usa `mode='EV'`; as 652 linhas do painel 2025 ficavam sem carregadores. | Aceitação explícita de `EV` e `EVSE`. A ausência caiu para 269 linhas, que refletem cobertura real. |
| **HIGH** | Fallback por componentes inoperante no IEA 2026 | Quando o agregado `powertrain='EV'` não existia, o código não retornava os componentes. | Fallback por chave dimensional, sem misturar agregado e componentes. |
| **HIGH** | Imputação silenciosa no `DataCleaner` | Vendas ausentes eram interpoladas e campos opcionais recebiam mediana/zero. | Limpeza agora valida e preserva missingness. |
| **MEDIUM** | Métrica derivada de infraestrutura usava fluxo de vendas | `ev_sales / (charging_points + 1)` era interpretada como EVs por carregador. | Quando disponíveis, usa `ev_stock / charging_points`; a razão oficial IEA continua preferível. |
| **MEDIUM** | Feature de CAGR usava a amostra inteira em todos os anos | O valor constante carregava informação futura para linhas históricas. | CAGR passou a ser expansivo, calculado apenas até cada data. |

## Metodologia estatística e modelos quantitativos

### Difusão

O Bass foi reforçado com validação de entradas finitas, tempo estritamente crescente, adoção cumulativa não decrescente e limites `M > max(S)`, `p > 0` e `q ≥ 0`. O cálculo evita a razão indefinida `q/p` quando `p` é inválido e salva SSE/RMSE após o ajuste. Os chamadores da aplicação passaram a usar tempo iniciado em 1, evitando a incompatibilidade entre uma primeira venda acumulada positiva e a condição matemática `N(0)=0` da forma fechada.

Os modelos Logistic e Gompertz continuam úteis como aproximações de curva, mas não possuem uma interpretação econômica automática para `K`/`M`. AIC, BIC, RMSE e MAPE exibidos no dashboard são métricas **in-sample**. Não devem ser interpretados como prova de desempenho preditivo.

| Modelo | Estado após auditoria | Risco remanescente |
|---|---|---|
| Bass | Parâmetros limitados, entrada validada, diagnóstico de ajuste. | O teto de mercado é estimado pela curva; falta um prior ou limite econômico externo. |
| Logistic | Ajuste limitado a taxa positiva. | Sem validação out-of-sample e sem validação econômica de capacidade de carga. |
| Gompertz | Ajuste limitado a taxa positiva. | Mesmo risco de extrapolação e falta de validação temporal. |

### Previsão e Monte Carlo

O GP agora valida arrays, torna a tendência opcional, limita a otimização logística de tendência e fornece `sample_paths`, que usa a covariância posterior para gerar trajetórias correlacionadas. O simulador Monte Carlo usa RNG local opcionalmente semeado, preferindo `sample_paths` quando disponível. Isso elimina a hipótese anterior de independência entre todos os anos futuros.

Os intervalos mostrados na interface foram renomeados como **intervalos empíricos**. Eles são quantis de trajetórias dependentes de modelo e de cenário, e não intervalos de confiança ou credibilidade por si sós. O motor condicionado a regimes agora calcula o primeiro incremento futuro em relação ao último nível acumulado, evitando o valor determinístico zero que existia no primeiro ano previsto.

| Severidade | Achado | Correção aplicada |
|---|---|---|
| **HIGH** | Monte Carlo utilizava normais independentes por horizonte e estado aleatório global. | Trajetórias correlacionadas via covariância GP e `default_rng(seed)`. |
| **HIGH** | O primeiro ano anual do regime forecast era forçado a zero por `diff(..., prepend=cum_sim[0])`. | Diferença calculada a partir do último nível histórico simulado. |
| **MEDIUM** | Ruído normal multiplicativo podia gerar vendas negativas. | Ruído lognormal e clipping de incrementos não negativos. |
| **MEDIUM** | “95% CI” e “80% CI” eram rótulos excessivos para quantis de simulação. | Interface e narrativa usam “empirical interval”/“empirical simulation quantiles”. |

### Regimes e pontos de mudança

O objeto chamado `HMMRegimeDetector` emprega um Gaussian Mixture Model, não um HMM. O código e a tela agora descrevem essa realidade. Componentes são ordenados pela primeira feature, as probabilidades são posteriores do GMM e a matriz de transição é uma frequência empírica de estados adjacentes; ela não é uma matriz de transição latente estimada por um HMM. A escolha de quatro regimes ainda é uma hipótese configurável, não uma conclusão selecionada por BIC/AIC em produção.

O `BayesianChangePoint` é mantido por compatibilidade pública, mas foi documentado e exibido como uma triagem heurística de deslocamento de média. Seus scores normalizados não devem ser lidos como probabilidades posteriores ou atribuições de causalidade econômica.

## EVQTI, Factor Engine e índice

O EVQTI mantém features defasadas e target de crescimento futuro, o que é uma boa separação temporal de primeira ordem. A validação integrada com o painel IEA 2026 processado produziu 59 países ranqueados, `rank_ic` médio de aproximadamente 0,233 e Information Ratio de aproximadamente 1,04. Esses números são diagnósticos locais da execução auditada; não devem substituir um protocolo pré-especificado de seleção e teste.

Foi corrigida uma inconsistência concreta no `FactorEngine`: `get_evqti_scores()` renomeava o score multifatorial para `evqti_score`, apesar de a API prometer compatibilidade baseada apenas nos fatores de penetração. Agora a API retorna `evqti_compat_score`.

| Severidade | Risco remanescente | Consequência |
|---|---|---|
| **MEDIUM** | O `FactorEngine` exige um cutoff temporal explícito para cada protocolo histórico. | O padrão usa o ano anterior ao último ano do painel; sem cutoff por origem, um backtest pode ainda não representar um protocolo walk-forward completo. |
| **MEDIUM** | Ainda não há um harness formal para avaliar pesos multifatoriais em holdout por origem temporal. | A performance do índice multifatorial continua sem uma estimativa completa e automatizada out-of-sample. |
| **MEDIUM** | Min–max para 0–100 é relativo ao cross-section disponível. | Scores não são comparáveis entre universos ou datas sem preservar a referência de escala. |
| **MEDIUM** | Fallback macroeconômico neutro no ETII pode alterar o ranking quando a API externa falha. | A indisponibilidade da fonte deve ser visível no dashboard e nas exportações. |

## Aplicação, desempenho, reprodutibilidade e segurança

A aplicação iniciou em modo headless com sucesso. O `st.cache_data` permanece no carregamento principal, e os objetos aleatórios dos módulos refatorados usam sementes locais quando fornecidas. A validação integrada percorreu `2025 Cars`, `2026 Cars`, `Combined Cars`, `2026 Total`, feature engineering, Bass/Logistic/Gompertz, GP, Monte Carlo, regime forecast e EVQTI com dados reais locais.

A revisão estática não encontrou ciclos de importação locais ou literais que parecessem credenciais embutidas. Não foi executada uma varredura de vulnerabilidades das dependências nem uma análise de secret scanning histórico; portanto, a conclusão de segurança é limitada ao código presente e à inspeção estática. O World Bank Indicators API é uma fonte externa sem chave para consulta, mas suas séries podem ter defasagem de publicação e devem ser tratadas como dados disponíveis somente após sua data de divulgação. [2]

| Verificação | Resultado |
|---|---|
| `pytest -q` antes das alterações | 31 aprovados |
| `pytest -q` depois das alterações | **37 aprovados** |
| Validação de dados IEA/painéis | Chaves únicas, shares 0–100 e unidades centrais consistentes |
| Validação integrada | Aprovada para quatro variantes de painel e modelos principais |
| Inicialização Streamlit headless | Aprovada |
| Mypy no repositório inteiro | Não concluído: excedeu a janela de validação devido ao escopo/imports. |
| Ruff direcionado | 73 achados, predominantemente imports não usados e blocos `except: pass`; nenhum erro de sintaxe selecionado. |

## Refatoração realizada

A refatoração foi deliberadamente limitada aos problemas que alteravam dados, resultados numéricos, reprodutibilidade ou honestidade metodológica. Não foram introduzidos frameworks, camadas artificiais ou mudanças de nomes públicos desnecessárias.

| Arquivo ou área | Alteração | Justificativa | Validação |
|---|---|---|---|
| `ingestion/aggregator.py` | Agregação canônica e fallback seguro por componentes. | Corrigir double counting e perda de infraestrutura IEA 2025. | Testes de agregação e validação de dados reais. |
| `ingestion/dataset_engine.py` | Fachada sobre `IEALoader`. | Eliminar semântica paralela no caminho da aplicação. | Validação integrada dos quatro painéis. |
| `preprocessing/cleaning.py` e `features.py` | Sem imputação silenciosa; features temporalmente ordenadas e unidades corrigidas. | Preservar integridade e evitar leakage. | Testes de propriedade e pipeline real. |
| `models/diffusion/bass.py` e chamadores | Restrições, estabilidade e eixo temporal. | Evitar ajustes inválidos e condição inicial incoerente. | Testes quantitativos e fit real. |
| `models/forecasting/*` | GP correlacionado, RNG local e quantis honestos. | Melhorar reprodutibilidade e semântica de incerteza. | Testes de reprodutibilidade e forecast real. |
| `models/regimes/*` e página | GMM/mean shift descritos corretamente. | Evitar apresentar proxies como HMM/Bayesianos. | Testes existentes e inicialização da aplicação. |
| `models/scoring/factor_engine.py` | Output EVQTI compatível corrigido e cutoff temporal para calibração. | Evitar expor score multifatorial sob a API do EVQTI e reduzir leakage retrospectivo. | Suíte completa e teste temporal dedicado. |

## Riscos remanescentes e próximos passos

A prioridade recomendada é construir um protocolo de validação temporal antes de ampliar telas ou complexidade. Para cada modelo de forecast e para o `FactorEngine`, defina datas de origem, congele winsorização/pesos usando apenas o treino, avalie walk-forward e reporte MAE/RMSE/sMAPE por país e agregado. A seleção de número de regimes deve comparar pelo menos K=2–5 com BIC/AIC e estabilidade de cluster, mantendo a interpretação de regimes como descritiva enquanto não houver dinâmica temporal explícita.

A camada de qualidade precisa de uma configuração de lint objetiva, com regras graduais e exclusões justificadas, seguida da remoção dos 73 achados direcionados. `except: pass` deve ser eliminado ou substituído por tratamento explícito e logging contextual. Um `pyproject.toml` com configurações de Ruff, Pytest e Mypy reduziria ambiguidade e melhoraria a reprodutibilidade do CI.

A documentação pública deve separar claramente dados históricos de projeções IEA, data de observação de data de disponibilidade e quantitativos observados de resultados modelados. As páginas de forecast e cenário devem exibir as hipóteses de distribuição/prior e um aviso sobre inexistência de inferência causal.

## Notas finais

| Dimensão | Nota final (0–10) | Fundamentação resumida |
|---|---:|---|
| Arquitetura | 7,0 | Pipeline de dados consolidado; ainda há lógica de apresentação e analytics distribuída. |
| Qualidade de código | 6,0 | Núcleo crítico melhorou, mas há 73 achados de lint direcionado. |
| Data Engineering | 7,5 | Chaves, agregação e missingness agora são mais rigorosos; cobertura segue incompleta. |
| Correção estatística | 6,5 | Restrições e terminologia melhoraram; faltam validações temporais formais. |
| Metodologia quantitativa | 6,0 | EVQTI tem estrutura temporal básica, mas o multifator ainda pode olhar o futuro. |
| Forecasting | 6,0 | GP/MC e intervalos melhoraram; extrapolação e avaliação out-of-sample permanecem fracas. |
| Machine Learning | 5,0 | GMM foi tornado honesto, porém K e estabilidade não são selecionados sistematicamente. |
| Estabilidade numérica | 7,0 | Guardas de domínio e jitter foram adicionados; modelos de curva ainda exigem sensibilidade. |
| Reprodutibilidade | 7,0 | Seeds locais e dados versionados; não há CI/configuração de tipos consolidada. |
| Testes | 6,5 | 37 testes cobrindo pontos críticos; cobertura de páginas e backtests é limitada. |
| Desempenho | 6,5 | Cache principal existe; falta profiling de páginas e custo de modelos por interação. |
| Segurança | 7,0 | Sem credenciais detectadas estaticamente; falta análise de dependências e secret scan histórico. |
| Streamlit | 6,5 | Inicialização validada; ainda há lógica quantitativa distribuída em páginas. |
| Documentação | 7,0 | Atualizada para refletir limites principais; requer manutenção junto de benchmarks futuros. |
| Qualidade de pesquisa | 6,0 | Fontes são adequadas, mas causalidade e previsão exigem governança temporal mais forte. |
| **Nota geral** | **6,5** | Projeto promissor e agora mais defensável, mas ainda não deve reivindicar validação preditiva ou causal robusta. |

## Referências

[1]: https://www.iea.org/data-and-statistics/data-tools/global-ev-data-explorer "International Energy Agency — Global EV Data Explorer"
[2]: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation "World Bank — Indicators API documentation"
