# ETIL — Research Evidence Index

**Status:** first evidence inventory  
**Inventory date:** 2026-11-27  
**Repository:** `edu-moraess/EV-Transition-Intelligence-Lab-ETIL-`  
**Validation baseline:** `3206e14b13db6a28f432573e3b874d678877449e`

This index maps the principal research, data, engineering, and validation evidence currently identified in the repository. It distinguishes existing documentary evidence from checks that were newly executed and from evidence still to be collected.

## 1. Project identity and development history

| Evidence | Location | What it supports | Status |
|---|---|---|---|
| Project overview, scope, research question, history and architecture | [README.md](../README.md) | ETIL's identity as an independent applied research project; documented repository history and project boundaries. | Existing |
| First public repository commit | [Commit 9ed782f](https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-/commit/9ed782f139547555b7966339598b8531893e3cf9) | Public repository history beginning 2026-06-16. This does not rule out earlier private study or prototypes. | Existing |
| Research integration | [PR #1](https://github.com/edu-moraess/EV-Transition-Intelligence-Lab-ETIL-/pull/1) | Integration of EVQTI temporal integrity and robustness work into `main`. | Merged |

## 2. Data and provenance

| Evidence | Location | What it supports | Status |
|---|---|---|---|
| Data dictionary | [docs/data_dictionary.md](data_dictionary.md) | Canonical panel fields, units, dimensions and missingness semantics. | Existing |
| Data source description | [README — Data](../README.md#3-dados-e-cobertura) | IEA Global EV Data Explorer as the principal source and World Bank indicators as optional context. | Existing |
| Raw dataset snapshots and cryptographic hashes | To be catalogued | Exact source files, retrieval dates, checksums, licensing and transformations for each experiment. | Pending inventory |

## 3. Methods and model implementation

| Evidence | Location | What it supports | Status |
|---|---|---|---|
| Methodology | [docs/methodology.md](methodology.md) | Diffusion models, GMM regime description, change-point terminology, temporal availability and EVQTI rolling-origin protocol. | Updated in this evidence branch |
| EVQTI implementation | [models/scoring/evqti.py](../models/scoring/evqti.py) | Fixed reference weights, lagged features, raw next-year target, train-only rolling-origin calibration and documented vintage limitation. | Existing code |
| Regime implementation | [models/regimes/hmm.py](../models/regimes/hmm.py) and [models/regimes/changepoint.py](../models/regimes/changepoint.py) | GMM implementation behind the legacy HMM class name and heuristic mean-shift implementation behind the legacy Bayesian class name. | Existing code |

## 4. Research experiments and results

| Evidence | Location | What it supports | Status |
|---|---|---|---|
| P1.7–P1.10 validation report | [docs/EVQTI_RESEARCH_VALIDATION_P1_7_P1_10.md](EVQTI_RESEARCH_VALIDATION_P1_7_P1_10.md) | Rolling-origin results, weight sensitivity, country-cluster bootstrap, leave-one-country-out and balanced-panel robustness. | Existing report |
| README results summary | [README — EVQTI results](../README.md#6-resultados-da-validação-evqti--p1) | Condensed account of the reported P1 experiments and their limitations. | Existing |
| Experiment source tables, figures, PDF and execution metadata | To be catalogued from their storage location | Traceability from each reported number to its generated artifact and run. | Pending inventory |
| Independent rerun of P1.7–P1.10 at the current main SHA | Not yet recorded | Re-execution of research experiments, distinct from running the unit/regression test suite. | Pending |

## 5. Software tests and reproducibility

### Post-merge test execution

| Field | Recorded value |
|---|---|
| Branch checked | `origin/main` |
| Remote main SHA | `3206e14b13db6a28f432573e3b874d678877449e` |
| Tested checkout SHA | `3206e14b13db6a28f432573e3b874d678877449e` |
| Command | `python -m pytest -q` |
| Result | **55 passed in 8.00s** |
| Process exit code | `0` |
| Run identifier | `20261127T012623Z` |
| Google Drive evidence folder | `MyDrive/ETIL_RESEARCH_EVIDENCE/main_validation/20261127T012623Z` |

The Drive run folder was created by the reproducibility cell and is expected to contain clone/fetch/checkout logs, Git status, dependency installation log, package environment, full pytest output and `summary.json`. The test result confirms the suite at the stated commit; it does not mean the P1 research experiments were rerun.

### Test source files identified

- `tests/test_evqti.py` — EVQTI behavior and temporal/anti-leakage regression tests.
- `tests/test_temporal_integrity.py` — temporal availability and calibration cutoffs.
- `tests/test_loader.py` — source schema and loader behavior.
- `tests/test_aggregator.py` — dimensional aggregation and mode filtering.
- `tests/test_corrections.py` — regression coverage for prior data/quantitative corrections.
- `tests/test_quantitative_guards.py` — numerical and simulation safeguards.

## 6. Engineering and application

| Evidence | Location | What it supports | Status |
|---|---|---|---|
| Streamlit entry point | `app.py` | Application entry point. | Present in repository |
| Streamlit pages | `streamlit_pages/` | User-facing analytical and research interface. | Present in repository |
| Dependencies | [requirements.txt](../requirements.txt) | Declared package requirements. Version ranges are not a fully pinned lockfile. | Existing |
| Historical engineering audit | [TECHNICAL_AUDIT_REPORT.md](../TECHNICAL_AUDIT_REPORT.md) | Snapshot of an earlier technical audit; must be labelled historical, not treated as a current audit. | Existing, historical |
| Current Streamlit runtime capture | To be collected | Screenshot, startup log, app version and data state for a reproducible interface demonstration. | Pending |

## 7. Known limitations and evidence gaps

1. The IEA historical release-vintage and publication-timestamp history is not available in the current panel; the walk-forward protocol is not fully vintage-aware.
2. The current post-merge evidence confirms the 55-test software suite, not a fresh rerun of P1.7–P1.10.
3. The current evidence run did not generate a coverage report or JUnit XML; neither should be claimed.
4. `requirements.txt` uses version ranges. A fully pinned environment lock and platform details remain to be archived.
5. Raw dataset files, retrieval metadata, checksums and licenses still need a systematic inventory.
6. A current Streamlit runtime capture has not yet been attached to this evidence index.
7. The August 2026 technical audit is historical and should not be presented as a November 2026 audit.

## 8. Next evidence-collection sequence

- **E1 — Data provenance:** inventory raw IEA/World Bank files, source URLs, retrieval dates, schemas, licenses and SHA-256 hashes.
- **E2 — Experiment artifacts:** map each P1.7–P1.10 table/figure/PDF back to its source data, script/notebook, run metadata and output hashes.
- **E3 — Application evidence:** capture Streamlit startup and representative screens, with the exact commit and data state.
- **E4 — Reproduction package:** preserve the tested Python/package environment and add a pinned lockfile only after choosing a supported reproducible environment.
- **E5 — Research validation:** rerun P1.7–P1.10 at a recorded commit and compare generated outputs with the report, without silently overwriting prior results.

---

**Interpretation rule:** an item marked *Existing* means the referenced repository artifact exists; it does not imply that every underlying experiment has been independently reproduced in the current environment. A *Pending* item is an explicit evidence gap, not a negative research result.
