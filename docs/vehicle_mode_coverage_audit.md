# ETIL — Vehicle-Mode Coverage Audit

**Date:** 2026-11-12  
**Scope:** Read-only repository and source-definition inspection  
**Status:** Partial verification; raw-row counts not yet verified

## 1. Repository data assets

The `main` branch contains the following raw IEA files under `data/raw/`:

| File | Repository size | Git blob SHA |
|---|---:|---|
| `EV_Data_Explorer_2025_GEVO_clean.csv` | 1,516,250 bytes | `0d06b919a94f33003d6556a1a7617fecd3b00d62` |
| `EV_Data_Explorer_2026_GEVO_clean.csv` | 4,719,544 bytes | `88d9b16ef00142679e4f5183904c03b58a126d39` |

The files are present in the repository. The GitHub file-content reader did not return CSV row content, so this inspection did not calculate actual mode counts, country coverage, year ranges or null rates from the raw files. Presence of a file is not evidence that every expected mode is populated.

## 2. Verified code-level support

Inspection of `ingestion/loader.py` on `main` confirms:
- `_VALID_MODES` includes `2 and 3 wheelers`.
- `load_dataset(..., mode=...)` accepts `2 and 3 wheelers` as a documented vehicle mode.
- `available_modes(year)` dynamically intersects observed `EV sales` mode values with the supported vehicle-mode set. This function is the correct entry point for discovering which vehicle modes actually exist in each raw release.
- The loader supports separate IEA 2025 and IEA 2026 schemas.

Inspection of `ingestion/aggregator.py` confirms:
- `_VEHICLE_MODES` includes `2 and 3 wheelers`.
- Sales and stock aggregation accept a mode argument.
- `total_ev_all_modes()` and `total_ev_stock_all_modes()` sum across selected modes. These totals must not replace mode-specific panels or be used to calculate cross-mode shares without an appropriate denominator.

Inspection of `app.py` confirms a visible mode selector with `Cars`, `2 and 3 wheelers`, and `Total (all modes)`.

These are verified code capabilities, not verified raw-data row counts.

## 3. IEA vehicle definitions

The IEA Global EV Data Explorer identifies these vehicle modes: Cars; 2 and 3 wheelers; Vans; Buses; Trucks.

The IEA describes 2- and 3-wheelers as motorcycles, scooters and rickshaws. Its stated scope excludes e-bikes and low-speed scooters; three-wheelers follow UNECE L2/L4/L5 classes.

The IEA mode list does not provide a separate quadricycle mode. Quadricycles (UNECE L6/L7) must therefore not be silently assigned to Cars or 2 and 3 wheelers. Any such mapping requires explicit source-methodology evidence. Otherwise ETIL needs a separate source and a separately identified category.

## 4. Important current limitations

- The current scoring modules are EV-oriented and their documented factor definitions are not a universal model for all vehicle modes.
- The default loader mode is `Cars`; this means a successful default pipeline run does not prove that 2/3-wheelers are analyzed.
- The current canonical `build_panel()` joins charging-point observations into a vehicle-mode panel. Charging-point metrics must be reviewed before being reused for 2/3-wheelers; the denominator and infrastructure coverage may not be comparable.
- A combined total across cars, two-/three-wheelers, vans, buses and trucks is a vehicle-count total, not a comparable market share, adoption rate or technology score.
- Quadricycles are not independently represented by the current mode taxonomy.

## 5. Required raw-data validation

Run a read-only audit against the checked-in CSVs before changing ingestion code. For each release (2025 and 2026), report:

1. Distinct `mode` values for `parameter == 'EV sales'` and `parameter == 'EV stock'`.
2. Row counts and distinct countries for each vehicle mode and parameter.
3. Historical year ranges and projection categories/year ranges per mode.
4. Powertrain values by mode and release.
5. Missing-value counts, duplicate dimensional keys and units.
6. Countries with sales but no stock, and vice versa.
7. Whether the mode is present in both releases and whether its coverage changes.
8. Whether sales shares and stock shares exist for each mode.
9. Whether 2/3-wheelers have any dedicated infrastructure measures in the raw files.
10. Whether any source metadata explicitly identifies quadricycles.

The audit must preserve source files and print file hashes. It must not rewrite CSVs, train models, install packages or modify repository state.

## 6. Current conclusion

**Code-level support for the `2 and 3 wheelers` mode: verified.**  
**Presence of the raw IEA files: verified.**  
**Actual raw-row coverage and counts for 2/3-wheelers in each committed CSV: not yet verified.**  
**Dedicated quadricycle coverage in the current IEA files: not established; no dedicated mode is exposed by the inspected taxonomy.**

Do not claim full dataset coverage until the raw-data validation above has been executed.