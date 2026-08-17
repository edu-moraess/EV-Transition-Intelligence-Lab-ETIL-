# Dicionário de Dados

## Painel canônico por modo

| Coluna | Descrição |
|---|---|
| `country` | Nome padronizado do país/economia. Agregados regionais são removidos quando `countries_only=True`. |
| `category` | Categoria IEA, normalmente `Historical` no painel usado pelo app; releases também podem conter cenários projetados. |
| `year` | Ano da observação. |
| `mode` | Modo de veículo (`Cars`, `2 and 3 wheelers` ou `Total`). |
| `ev_sales` | Vendas anuais de veículos elétricos, em veículos. |
| `ev_stock` | Estoque acumulado de veículos elétricos, em veículos; pode ser ausente quando a fonte não publica o valor. |
| `ev_sales_share` | Participação percentual de EV nas vendas do modo. |
| `ev_stock_share` | Participação percentual de EV no estoque do modo. |
| `charging_points` | Número de pontos públicos de recarga; cobertura depende do release e do país. |
| `data_source` | Release ou combinação que originou a linha (`2025`, `2026`, `combined` ou schema detectado). |

## Campos do modo `Total`

O modo `Total` agrega todos os modos de veículo disponíveis no release. Como a fonte local publica shares por modo, os shares totais são reconstruídos como razão entre a soma de EVs e os denominadores implícitos de cada modo (`EV value / mode share`). O resultado é mantido como ausente quando algum modo com valor EV não possui share válido.

| Coluna | Descrição |
|---|---|
| `share_coverage_sales` | Proporção de modos com share de vendas válido no cálculo total. |
| `share_coverage_stock` | Proporção de modos com share de estoque válido no cálculo total. |
| `share_coverage` | Mínimo entre cobertura de vendas e de estoque; indica a cobertura conjunta do total. |

## Features derivadas

| Coluna | Descrição |
|---|---|
| `ev_sales_log` | Log natural de `ev_sales` após tratamento de zeros como ausentes para a transformação logarítmica. |
| `penetration` | Share/estoque de EV usado pelo pipeline quando o denominador está disponível. |
| `growth_rate` | Crescimento percentual anual de `ev_sales`, calculado por país e em ordem temporal. |
| `cagr` | CAGR expansivo calculado somente com o histórico disponível até a linha. |
| `momentum_3y` | Crescimento/momentum de janela de três anos, sujeito à disponibilidade de observações. |
| `volatility` | Volatilidade calculada sobre a série de crescimento disponível. |
| `ev_per_charger` | Razão de estoque EV por ponto público quando ambos os valores existem. |
| `ev_per_capita` | Estoque EV por população quando a população está disponível. |

Valores ausentes são preservados e não devem ser interpretados automaticamente como zero. As colunas `total_sales`, `gdp` e `population` podem existir em fontes auxiliares ou em painéis enriquecidos, mas não são garantidas pelo contrato mínimo do painel IEA.
