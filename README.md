# EV Transition Intelligence Lab — ETIL

**Independent Applied Research Project | Engenharia de dados, modelagem quantitativa e pesquisa aplicada sobre a transição global para veículos elétricos.**

O **EV Transition Intelligence Lab (ETIL)** é um projeto independente de pesquisa aplicada iniciado em **16 de junho de 2026**, conforme o registro de criação do repositório e seu primeiro commit público. O projeto investiga padrões de adoção de veículos elétricos (EVs), diferenças entre trajetórias nacionais, regimes de transição e associações entre penetração atual e crescimento futuro.

O ETIL combina engenharia de software, engenharia de dados, análise quantitativa, estatística e modelagem matemática. Sua aplicação Streamlit é uma interface para exploração e experimentação; o objetivo de pesquisa vai além do dashboard: formular hipóteses, construir indicadores, avaliar modelos, testar robustez e documentar limitações.

> **Natureza e estágio:** pesquisa independente em desenvolvimento, com experimentos quantitativos e validações metodológicas documentadas. Não é pesquisa institucional, artigo revisado por pares ou resultado acadêmico formalmente publicado. Os resultados são exploratórios e condicionados aos dados e procedimentos descritos neste repositório.

- **Repositório:** [edu-moraess/EV-Transition-Intelligence-Lab-ETIL-](https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-)
- **Início documentado:** 16/06/2026.
- **Domínio:** transição energética, mobilidade elétrica, adoção tecnológica e análise comparativa internacional.
- **Status:** pesquisa aplicada independente em evolução.

## 1. Pergunta e objetivo de pesquisa

O ETIL investiga como a adoção de veículos elétricos evolui entre países e se indicadores defasados de penetração — participação de EVs no estoque e nas vendas — contêm informação associativa sobre o crescimento subsequente das vendas.

Uma linha central de investigação é a hipótese de que países em estágios relativamente iniciais de adoção podem apresentar crescimento futuro proporcionalmente maior. Essa hipótese é testada por meio de indicadores defasados, ordenação cross-sectional e avaliações temporais. Ela não é tratada como causalidade nem como regra universal.

Objetivos complementares:

- organizar e validar dados internacionais de adoção de EVs;
- caracterizar trajetórias históricas e regimes de adoção;
- comparar modelos de difusão tecnológica;
- explorar cenários condicionais e incerteza de simulação;
- construir indicadores comparativos interpretáveis;
- avaliar estabilidade, sensibilidade e robustez dos resultados;
- tornar explícitas as limitações dos dados, modelos e inferências.

## 2. Histórico e evolução

O histórico público do GitHub documenta o desenvolvimento desde junho de 2026. A data abaixo é a primeira atividade documentada no repositório atual; não pretende afirmar que ideias, estudos ou protótipos pessoais não tenham existido antes dela.

| Período | Evolução documentada |
|---|---|
| **16/06/2026 — Fundação** | Criação do repositório e início da aplicação ETIL. Os primeiros commits adicionaram a aplicação, dados IEA, modelos de difusão Bass/Logistic/Gompertz, Forecast Lab com Gaussian Process e Monte Carlo e a primeira implementação do EVQTI. |
| **Junho de 2026 — Expansão** | Ampliação dos módulos de dados, indicadores, análise de países, cenários, infraestrutura e visualização. |
| **13/08/2026 — Plataforma de pesquisa quantitativa** | Reorganização registrada no commit `33693d7`, com a mensagem “transform dashboard into quant research platform”. |
| **Agosto–outubro de 2026 — Integridade e consolidação** | Evolução da ingestão canônica, processamento temporal, documentação de limitações e testes do núcleo analítico. |
| **Novembro de 2026 — Validação EVQTI P1** | Avaliações rolling-origin, sensibilidade de pesos, bootstrap por país, leave-one-country-out e painel balanceado. |
| **27/11/2026 — Consolidação na main** | Integração da linha de pesquisa EVQTI por meio do PR #1, merge commit `7ccc731`. |

### Marcos públicos de referência

- [Primeiro commit do repositório — 16/06/2026](https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-/commit/9ed782f139547555b7966339598b8531893e3cf9)
- [Primeira aplicação ETIL](https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-/commit/3000522773d5b905e945ba6269655eca50f127cc)
- [Inclusão dos dados IEA](https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-/commit/39aba4edc08eb9a1aa806635ed92a814b595d465)
- [Primeiro Diffusion Engine](https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-/commit/282f525a2f02eda82c001e50e58ac2ccf3b94a1f)
- [Primeiro Forecast Lab](https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-/commit/2fa8f6e7cffef062b4a99366dc1e2ada5a521295)
- [Primeira implementação do EVQTI](https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-/commit/6065c134908130c49938221ef561e78617fd30e8)
- [Transformação em plataforma de pesquisa quantitativa — 13/08/2026](https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-/commit/33693d72a22455ff2711de5ab5ccf1bd289ec32f)
- [PR #1 — integração da pesquisa EVQTI](https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-/pull/1)

## 3. Dados e cobertura

A fonte principal é o [Global EV Data Explorer da International Energy Agency (IEA)][1], que disponibiliza séries históricas e projeções de vendas, estoque e infraestrutura de recarga. Indicadores macroeconômicos opcionais são obtidos pela [World Bank Indicators API][2].

| Painel processado — Cars | Países | Período | Observações |
|---|---:|---:|---|
| IEA 2025 | 53 | 2010–2024 | Cobertura parcial conforme indicador e país. |
| IEA 2026 | 60 | 2010–2025 | Release mais recente utilizado na consolidação. |
| Combined | 60 | 2010–2025 | Valores IEA 2026 prevalecem nas chaves país-ano sobrepostas. |

A validação P1 do EVQTI utilizou o painel histórico IEA 2026 — Cars: 60 países e anos de 2010 a 2025. A avaliação rolling-origin reportada cobre nove anos-alvo, de 2016 a 2024. O painel balanceado dessa análise contém 48 países com cobertura comum nas nove janelas.

Arquivos locais esperados:

```text
data/raw/EV_Data_Explorer_2025_GEVO_clean.csv
data/raw/EV_Data_Explorer_2026_GEVO_clean.csv
data/raw/fuel_prices_iea.csv
```

A disponibilidade de cada série depende do país, ano, categoria e release. Valores ausentes devem permanecer identificáveis; não são tratados como zero nem interpolados silenciosamente. Projeções publicadas pela IEA devem ser distinguidas de observações históricas.

## 4. Arquitetura de software e dados

A ingestão possui uma implementação canônica. `IEALoader` valida o arquivo de origem e `Aggregator` aplica seleção de powertrains sem misturar indevidamente o agregado `EV` com BEV/PHEV/FCEV/EREV. `DatasetEngine` funciona como fachada de compatibilidade para a aplicação.

```text
IEA raw data / World Bank indicators
        ↓
IEALoader / Aggregator
        ↓
DataCleaner
        ↓
DataTransformer
        ↓
FeatureEngineer
        ↓
models / analytics / scoring
        ↓
Streamlit application
```

| Camada | Componentes | Responsabilidade |
|---|---|---|
| Ingestão | `ingestion/loader.py`, `ingestion/aggregator.py` | Schema, chaves, agregação dimensional e filtros de regiões agregadas. |
| Processamento | `preprocessing/` | Tipos, validação país-ano e features cronologicamente ordenadas. |
| Difusão | `models/diffusion/` | Modelos de adoção cumulativa. |
| Forecasting | `models/forecasting/` | Gaussian Process e simulação Monte Carlo. |
| Regimes | `models/regimes/` | Classificação exploratória com Gaussian Mixture Model. |
| Índices | `models/scoring/` | EVQTI e ETII. |
| Aplicação | `app.py`, `streamlit_pages/` | Interface, visualização e orquestração. |

## 5. Métodos e componentes analíticos

### Modelos de difusão

O ETIL implementa curvas **Bass, Logistic e Gompertz** para descrever trajetórias de adoção cumulativa. No modelo Bass, são aplicadas restrições de domínio como (M > \max(adoption)), (p > 0) e (q \geq 0).

AIC, BIC, RMSE, MAPE e R² calculados sobre os dados usados no ajuste são métricas **in-sample**. Não constituem, por si só, evidência de capacidade preditiva fora da amostra.

### Forecast Lab

O Forecast Lab combina **Gaussian Process** com trajetórias **Monte Carlo** correlacionadas pela covariância posterior. As faixas exibidas são quantis empíricos das simulações condicionadas ao modelo, aos dados e às hipóteses do cenário. Não devem ser interpretadas automaticamente como intervalos de confiança ou de credibilidade.

### Regimes de adoção

A classificação utiliza **Gaussian Mixture Model (GMM)**. Os rótulos são ordenados com base em uma feature de adoção e têm finalidade descritiva. A classificação depende das features, da especificação e do número de componentes escolhido; não representa categorias econômicas naturais ou permanentes.

### EVQTI — EV Transition Quantitative Indicator

O **EVQTI** é uma linha de pesquisa do ETIL que investiga a associação entre penetração atual de EVs e crescimento futuro das vendas.

A especificação validada utiliza shares defasados:

- `ev_stock_share_lag1`: participação de EVs no estoque, defasada;
- `ev_sales_share_lag1`: participação de EVs nas vendas, defasada;
- target primário: `growth_t1`, crescimento futuro das vendas.

A hipótese direcional é que menor penetração defasada pode estar associada a maior crescimento subsequente. O score é construído a partir de sinais negativos dos shares defasados, com normalização cross-sectional. A especificação oficial utiliza pesos:

| Componente | Peso |
|---|---:|
| Stock share defasado | 0,485 |
| Sales share defasado | 0,515 |

Na avaliação rolling-origin, os parâmetros estimados em cada origem devem utilizar apenas informação disponível até aquela origem. O target bruto `growth_t1` é a métrica primária; a versão winsorizada é mantida como diagnóstico, não como substituta do resultado principal.

Os pesos oficiais 0,485/0,515 foram derivados da configuração histórica do painel completo e são tratados como especificação fixa de referência, não como pesos recalibrados point-in-time em cada origem. A ausência de vintages e timestamps históricos de publicação da IEA impede classificar o exercício como totalmente vintage-aware.

### ETII

O **ETII** combina o EVQTI com fundamentos macroeconômicos do World Bank quando há cobertura disponível. Diferenças de cobertura e falhas na API podem afetar comparabilidade; indisponibilidade de dados externos deve ser explicitada e não mascarada.

## 6. Resultados da validação EVQTI — P1

Os resultados abaixo correspondem aos experimentos documentados P1.7–P1.10. São evidências de consistência associativa dentro do painel e do desenho experimental utilizado, não prova de causalidade ou garantia de previsão futura.

### P1.7 — Rolling-origin

Painel histórico: 60 países. Anos-alvo: 2016–2024.

| Métrica | Resultado |
|---|---:|
| Mean annual Rank IC | **0,3131** |
| Mediana anual Rank IC | 0,2400 |
| Anos com Rank IC positivo | **9/9** |
| Mean top-bottom spread | 338,6684 |
| Peso médio stock | 0,5084 |
| Peso médio sales | 0,4916 |

Os nove Rank ICs anuais foram positivos. A magnitude variou entre anos, portanto a média não deve ser lida como desempenho uniforme.

### P1.8 — Sensibilidade dos pesos

Mean raw Rank IC por cenário:

| Cenário | Mean Rank IC | Anos positivos |
|---|---:|---:|
| Equal weight | 0,3157 | 9/9 |
| Official | **0,3165** | 9/9 |
| Sales 75% | 0,3242 | 9/9 |
| Sales only | 0,3291 | 9/9 |
| Stock 75% | 0,3066 | 9/9 |
| Stock only | 0,2797 | 8/9 |

Sales-only apresentou maior média nessa comparação, mas menor estabilidade de ranking entre anos do que algumas alternativas. Stock-only teve menor média de Rank IC. A especificação oficial foi mantida como compromisso de referência, não porque tenha vencido todos os cenários.

### P1.9 — Bootstrap e leave-one-country-out

Foram utilizadas 2.000 réplicas de cluster bootstrap por país.

| Cenário | Rank IC observado | Média bootstrap | IC bootstrap 95% |
|---|---:|---:|---:|
| Official | 0,3165 | 0,3096 | [0,2242; 0,3878] |
| Equal | 0,3157 | 0,3087 | [0,2244; 0,3911] |
| Sales only | 0,3291 | 0,3215 | [0,2322; 0,4105] |
| Stock only | 0,2797 | 0,2818 | [0,1946; 0,3613] |

Na análise leave-one-country-out, nenhuma exclusão individual dos 60 países tornou negativo o Rank IC médio. A cobertura foi de 48 países com nove anos completos, 52 com pelo menos sete anos e mediana de nove anos.

O bootstrap avalia sensibilidade à reamostragem cross-country condicionada ao painel. Não resolve dependência temporal, revisões históricas da IEA ou outros vieses de especificação.

### P1.10 — Painel balanceado

A análise foi repetida em 48 países comuns às nove janelas.

| Cenário | Painel completo | Painel balanceado | Diferença |
|---|---:|---:|---:|
| Equal | 0,3157 | 0,3137 | -0,0020 |
| Official | 0,3165 | **0,3149** | -0,0016 |
| Sales only | 0,3291 | 0,3253 | -0,0038 |
| Stock only | 0,2797 | 0,2899 | +0,0102 |

Todos os cenários tiveram Rank IC anual positivo em 9/9 anos no painel balanceado. Os intervalos bootstrap de 2.000 réplicas também ficaram acima de zero:

| Cenário | Mean Rank IC | IC bootstrap 95% |
|---|---:|---:|
| Official | 0,3149 | [0,2178; 0,3939] |
| Equal | 0,3137 | [0,2149; 0,4033] |
| Sales only | 0,3253 | [0,2329; 0,4053] |
| Stock only | 0,2899 | [0,1871; 0,3730] |

A proximidade entre painel completo e balanceado reduz a preocupação de que o sinal observado decorra exclusivamente da entrada e saída de países na amostra. Não elimina outras fontes de viés ou dependência.

### Síntese dos resultados

Nos experimentos P1 documentados, o EVQTI apresentou associação cross-sectional positiva entre score e crescimento futuro das vendas nas janelas avaliadas, com resultados relativamente estáveis a diferentes composições de amostra e cenários de pesos. A evidência é promissora, mas permanece exploratória e condicionada à especificação, à amostra e à fonte de dados.

## 7. Reprodutibilidade, testes e artefatos

A pesquisa foi conduzida em etapas versionadas, com relatórios e evidências associados a commits. Na validação pré-merge do EVQTI, a suíte registrou **55 testes aprovados** no commit `5223a9c`; uma suíte focada registrou 21 testes aprovados. O relatório P1 foi posteriormente integrado à branch `main` pelo PR #1.

**Nota de reprodutibilidade:** os 55 testes referem-se à execução registrada naquele commit de pesquisa, anterior ao merge final. Uma nova execução completa diretamente no merge commit `7ccc731` ainda deve ser registrada antes de afirmar que esse número foi reproduzido no estado atual da `main`.

O relatório visual P1 V2 contém gráficos de rolling-origin, sensibilidade de pesos, intervalos bootstrap e comparação do painel balanceado, além de tabelas CSV, metadados e PDF. A geração desses artefatos utilizou os resultados P1.7–P1.10 já calculados; **a geração do PDF não foi uma nova execução do backtest**.

## 8. Limitações científicas e de interpretação

As limitações são parte explícita da pesquisa:

1. **Revisões da IEA e vintages:** sem snapshots históricos e timestamps de publicação por observação, não é possível garantir uma avaliação totalmente point-in-time/vintage-aware.
2. **Dependência temporal e cross-country:** observações de países e anos podem não ser independentes. O bootstrap por país não captura toda a dependência temporal.
3. **Múltiplas comparações:** p-values nominais, quando apresentados, não corrigem automaticamente multiplicidade de testes.
4. **Amostra e cobertura:** países possuem disponibilidade desigual de séries e diferentes estágios de adoção.
5. **Associação não é causalidade:** Rank IC e spreads não identificam mecanismos causais.
6. **Generalização:** resultados em 2016–2024 não garantem comportamento em períodos futuros ou em outras fontes de dados.
7. **Métricas in-sample:** métricas de ajuste de curvas não substituem validação fora da amostra.
8. **Cenários condicionais:** simulações dependem das hipóteses e especificações adotadas.
9. **Regimes descritivos:** clusters dependem da seleção de features e hiperparâmetros.
10. **Fundamentos externos:** cobertura ou indisponibilidade do World Bank pode afetar o ETII.

O ETIL não deve ser usado como recomendação de investimento, previsão garantida, classificação causal de países ou instrumento de decisão pública sem validação adicional.

## 9. Próximas etapas de pesquisa

- **P2 — Reprodutibilidade pós-merge:** executar novamente a suíte e os experimentos no estado atual da `main`, registrar ambiente, versões, hashes e artefatos.
- **P3 — Baselines:** comparar EVQTI com indicadores simples de penetração de estoque e vendas, crescimento histórico e rankings alternativos.
- **P4 — Validação externa e temporal:** ampliar testes walk-forward, avaliar outros períodos e, quando possível, dados com vintages identificáveis.
- **P5 — Sensibilidade metodológica:** aprofundar especificações, cobertura, dependência temporal, múltiplas comparações e estabilidade.
- **P6 — Manuscrito:** consolidar pergunta, literatura relacionada, dados, metodologia, resultados, limitações e materiais de reprodução em um manuscrito científico.
- **P7 — Avaliação externa:** buscar revisão por pesquisadores independentes e eventual submissão a congresso ou periódico compatível.

Essas etapas são propostas de trabalho futuro, não resultados já concluídos.

## 10. Instalação e execução

Recomenda-se Python 3.11 ou superior e ambiente virtual isolado.

```bash
git clone https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-
cd EV-Transition-Intelligence-Lab-ETIL-
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Para executar os testes:

```bash
pytest -q
```

Os datasets IEA e eventuais arquivos locais devem ser obtidos de suas fontes e colocados nos caminhos esperados. Não inclua credenciais ou dados com restrições no repositório.

## 11. Licença e uso dos dados

O código do projeto é disponibilizado sob a licença MIT, conforme o arquivo [LICENSE](LICENSE). Os dados da IEA e as fontes externas mantêm seus próprios termos, licenças, condições de redistribuição e metodologias. Consulte as fontes originais antes de redistribuir datasets ou resultados derivados.

[1]: https://www.iea.org/data-and-statistics/data-tools/global-ev-data-explorer "International Energy Agency — Global EV Data Explorer"
[2]: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation "World Bank — Indicators API documentation"

---

**ETIL — Independent Applied Research Project**  
Pesquisa independente, desenvolvimento contínuo e validação metodológica documentada.
