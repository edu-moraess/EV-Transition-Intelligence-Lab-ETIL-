# EV Transition Intelligence Lab — ETIL

O **EV Transition Intelligence Lab (ETIL)** é uma aplicação Streamlit para explorar a transição de veículos elétricos por país, com dados IEA, métricas de adoção, curvas de difusão, classificação de regimes, cenários e índices de pesquisa. O projeto foi auditado tecnicamente em agosto de 2026; o relatório completo está em [TECHNICAL_AUDIT_REPORT.md](TECHNICAL_AUDIT_REPORT.md).

> **Escopo de pesquisa.** ETIL é uma ferramenta analítica e educacional. Seus rankings, cenários e previsões não constituem recomendação de investimento, inferência causal ou previsão garantida.

## Dados e cobertura

A fonte principal é o [Global EV Data Explorer da International Energy Agency][1], que disponibiliza histórico e projeções de vendas, estoque e infraestrutura de recarga de veículos elétricos. Indicadores macroeconômicos opcionais vêm da [World Bank Indicators API][2]. A disponibilidade de cada série depende do país, ano e release de origem; valores ausentes são preservados, não interpolados silenciosamente.

| Painel histórico processado | Países | Anos | Observações |
|---|---:|---:|---|
| IEA 2025 — Cars | 53 | 2010–2024 | O modo de carregadores usa `EV`; o painel preserva cobertura parcial. |
| IEA 2026 — Cars | 60 | 2010–2025 | O modo de carregadores usa `EVSE`; o release tem prioridade na combinação. |
| Combined — Cars | 60 | 2010–2025 | Observações IEA 2026 prevalecem nas chaves `country × year` sobrepostas. |

Os dados locais esperados são:

```text
data/raw/EV_Data_Explorer_2025_GEVO_clean.csv
data/raw/EV_Data_Explorer_2026_GEVO_clean.csv
data/raw/fuel_prices_iea.csv
```

## Arquitetura

A ingestão possui uma única implementação canônica. `IEALoader` valida o arquivo bruto e `Aggregator` aplica a seleção de powertrains sem misturar o agregado `EV` com BEV/PHEV/FCEV/EREV. `DatasetEngine` é mantido como uma fachada de compatibilidade para a aplicação.

```text
IEA raw data
  → IEALoader / Aggregator
  → DataCleaner
  → DataTransformer
  → FeatureEngineer
  → models / analytics
  → Streamlit pages
```

| Camada | Componentes | Responsabilidade |
|---|---|---|
| Ingestão | `ingestion/loader.py`, `ingestion/aggregator.py` | Schema, chaves, agregação dimensional e filtros de regiões agregadas. |
| Processamento | `preprocessing/` | Tipos, validação de país-ano e features ordenadas cronologicamente. |
| Modelos | `models/diffusion/`, `models/forecasting/`, `models/regimes/` | Curvas de adoção, GP, simulação e classificação de regimes. |
| Índices | `models/scoring/` | EVQTI e score multifatorial explicável. |
| Aplicação | `app.py`, `streamlit_pages/` | Interface, gráficos e orquestração dos módulos. |

## Modelos e interpretação

O ETIL disponibiliza Bass, Logistic e Gompertz para ajuste de adoção cumulativa. Bass valida tempo e adoção acumulada, impõe `M > max(adoption)`, `p > 0` e `q ≥ 0`, e retorna diagnósticos básicos de ajuste. AIC, BIC, RMSE, MAPE e R² apresentados na aplicação são **métricas de ajuste in-sample**; eles não demonstram desempenho preditivo fora da amostra.

O Forecast Lab combina um Gaussian Process com trajetórias Monte Carlo correlacionadas pela covariância posterior. As faixas apresentadas são **quantis empíricos de simulação condicionados ao modelo e ao cenário**. Elas não são automaticamente intervalos de confiança ou de credibilidade.

A classificação de regimes usa um **Gaussian Mixture Model (GMM)**. Os rótulos são ordenados pela primeira feature de adoção e servem à interpretação descritiva. A tela de rupturas usa uma triagem heurística de deslocamentos de média; os anos destacados não identificam causalidade econômica por si sós.

## EVQTI e ETII

O EVQTI é um fator de pesquisa que utiliza shares de estoque e vendas de EV defasados para avaliar associação com crescimento futuro de vendas. A construção mantém o target em `t+1` separado das features em `t-1`, mas o resultado ainda deve ser avaliado em janelas temporais pré-especificadas antes de qualquer uso decisório.

O ETII combina o EVQTI com fundamentos do World Bank quando a cobertura externa existe. As consultas macroeconômicas usam o ano máximo do painel EV como referência quando disponível. Uma falha da API reduz a confiabilidade comparativa do componente fundamental; esse fallback não deve ser ocultado ao interpretar rankings.

O modo `Total` agrega todos os modos de veículo e calcula shares ponderados pelos denominadores implícitos de cada modo. A cobertura de vendas e estoque é exposta nas colunas `share_coverage_sales`, `share_coverage_stock` e `share_coverage`.

## Instalação e execução

Use Python 3.11 ou superior e um ambiente virtual isolado.

```bash
git clone https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-
cd EV-Transition-Intelligence-Lab-ETIL-
pip install -r requirements.txt
streamlit run app.py
```

Para validar o núcleo do projeto:

```bash
pytest -q
```

A auditoria de agosto de 2026 adicionou testes para agregação de dados, missingness, modo Total, cutoff temporal do scoring, restrições do Bass, reprodutibilidade Monte Carlo e ordenação de quantis de regime. A suíte conta com 44 testes no estado atual.

## Limitações conhecidas e próximos passos

A validação temporal formal continua sendo uma prioridade metodológica. O `FactorEngine` agora usa, por padrão, o ano anterior ao último ano do painel como cutoff de calibração e aceita `calibration_end_year` explícito; backtests rigorosos ainda devem ajustar e congelar parâmetros em cada origem temporal. A escolha do número de regimes também deve comparar K=2–5 por BIC/AIC e estabilidade de cluster.

O teto de mercado dos modelos de difusão continua sendo estimado a partir da curva. Antes de tratar projeções como cenários de política ou de mercado, é necessário introduzir hipóteses econômicas externas e comparar sistematicamente erros walk-forward por país e horizonte.

## Licença e dados

O código é disponibilizado sob a licença MIT. O uso dos dados IEA e das fontes externas deve respeitar seus respectivos termos, licenças e atualizações de metodologia. [1] [2]

[1]: https://www.iea.org/data-and-statistics/data-tools/global-ev-data-explorer "International Energy Agency — Global EV Data Explorer"
[2]: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation "World Bank — Indicators API documentation"
