# EVQTI — protocolo de reprodução e proveniência

## Referência de código

- Repositório: edu-moraess/EV-Transition-Intelligence-Lab-ETIL-
- Branch: main
- Commit: 75e43af68f87d449bf83436976e856144059b4e6
- Implementação: models/scoring/evqti.py
- Loader: ingestion/loader.py
- Fonte: IEA Global EV Data Explorer 2026, Historical, Cars.

## Procedimento de reprodução

1. Obter os arquivos IEA 2026 diretamente da fonte autorizada e registrar nome, tamanho e SHA-256 de cada arquivo.
2. Fazer checkout do commit de referência acima.
3. Registrar versão do Python e dependências instaladas.
4. Carregar o painel com release 2026, categoria Historical, modo Cars e exclusão de entidades agregadas (countries_only=True).
5. Confirmar 60 países, 760 observações e anos 2010–2025 antes do backtest.
6. Executar EVQTI().backtest(panel, train_years=6, test_years=1, step=1, min_countries=10).
7. Confirmar nove janelas de 2016 a 2024 e calcular a média da coluna ic_mean.
8. Para comparação com baselines, restringir o painel à coorte de 45 países completos nas nove janelas e manter a mesma amostra para todos os sinais.
9. Para o bootstrap, executar 2.000 réplicas pareadas por país com seed 20261127, recalibrando o pipeline EVQTI em cada réplica e recalculando a normalização cross-sectional.
10. Para uma nova execução, arquivar os CSVs/JSON originais, parâmetros, ambiente, hashes dos dados e logs de execução junto ao relatório. Os artefatos da execução documentada em 27/11/2026 estão preservados em [EVQTI_RESEARCH_ARTIFACTS_2026-11-27.zip](artifacts/EVQTI_RESEARCH_ARTIFACTS_2026-11-27.zip), com manifesto SHA-256 e nota de reprodução.

## Integridade temporal da implementação

A função backtest documenta calibração com labels anteriores à origem de teste. Em cada origem, a calibração usa os seis anos anteriores. As features de score são shares defasados; o alvo futuro bruto não entra no score. A padronização cross-sectional do ano de teste utiliza apenas as features daquele ano.

A função _build_panel constrói lags por país e ano. Os limites de winsorização das features são estimados com valores históricos anteriores ao ano corrente. O alvo bruto growth_t1 é preservado para avaliação; growth_t1_win é uma série diagnóstica e não é usada como alvo principal do backtest.

## Limitações para reprodução independente

- A IEA pode revisar séries históricas entre releases.
- O repositório não contém necessariamente os arquivos de dados licenciados.
- O painel histórico final não permite reconstruir integralmente os vintages disponíveis em cada origem.
- Os resultados bootstrap são condicionais ao painel e ao procedimento de amostragem documentados.
- Os CSVs/JSON desta execução estão arquivados dentro do ZIP de artefatos; caminhos locais do Colab não são acessíveis a partir do GitHub.

## Checklist de arquivamento

- [x] Arquivar os CSVs/JSON originais de walk-forward, baseline e bootstrap no ZIP versionado.
- [x] Registrar SHA-256 de cada artefato no manifesto incluído no ZIP.
- [ ] Registrar versões de Python, pandas, numpy, scipy e scikit-learn.
- [ ] Registrar parâmetros e seed em JSON.
- [ ] Registrar log integral da execução e resumo de validação.
- [ ] Verificar que os arquivos correspondem à execução indicada, sem reconstrução manual.
