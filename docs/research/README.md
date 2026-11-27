# EVQTI — documentação de pesquisa

Esta pasta reúne os registros de pesquisa e reprodução associados à avaliação EVQTI concluída em 27/11/2026.

- [Manuscrito v1.0](EVQTI_MANUSCRIPT_v1.0.md): versão editorial consolidada, com metodologia, resultados, discussão, limitações e referências.
- [Registro experimental consolidado](EVQTI_EXPERIMENT_RECORD_2026-11-27.md): métricas rolling-origin, comparação com baselines, bootstrap pareado e interpretação dos resultados.
- [Protocolo de reprodução e proveniência](EVQTI_REPRODUCIBILITY.md): código de referência, parâmetros e checklist de arquivamento.
- [Pacote de artefatos experimentais](artifacts/EVQTI_RESEARCH_ARTIFACTS_2026-11-27.zip): nove CSVs/JSONs, manifesto SHA-256 e nota de proveniência da reprodução.

## Estado do pacote

- Manuscrito v1.0, resultados agregados e artefatos experimentais arquivados nesta branch.
- Os arquivos `EVQTI_WALK_FORWARD_MAIN_75e43af.csv` e `EVQTI_WALK_FORWARD_SUMMARY_75e43af.json` foram reproduzidos em 27/11/2026 no checkout `main`, commit `75e43af68f87d449bf83436976e856144059b4e6`; não são recuperações dos arquivos originais.
- O manifesto registra hashes SHA-256 dos nove artefatos; a nota de reprodução registra parâmetros, amostra e resultado.
- O nome por extenso do acrônimo ainda diverge entre o manuscrito (Electric Vehicle Transition Quality Index) e a docstring da implementação (EV Quant Transition Index). A nomenclatura foi preservada no manuscrito e a divergência está explicitada; harmonização exige decisão editorial explícita.
- Os arquivos estão em PR draft e ainda não foram integrados à main.

A documentação não substitui os dados de origem, os artefatos completos por réplica ou uma reprodução independente do experimento.
