# ETIL — Strategic Scope and Domain Architecture

**Status:** Proposed architecture baseline  
**Date:** 2026-11-12  
**Change type:** Documentation only; no runtime behavior changed.

## 1. Strategic identity

ETIL is an independent research platform for **Automotive, Mobility and Energy Transition Intelligence**.

Electric vehicles remain a core research domain, but they are not the full scope. ETIL studies how vehicle technologies, transport activity, energy systems, infrastructure, industrial supply chains, economics and public policy interact during the transition of mobility and energy.

ETIL is not intended to become an unrestricted general-purpose energy dashboard. Every domain must have a defined research question, measurable variables, documented sources, explicit coverage and a validation strategy.

## 2. Domain model

The platform is organized around connected research domains:

| Domain | Scope | Initial analytical questions |
|---|---|---|
| Automotive markets | Vehicle sales, stock, powertrains, segments, manufacturers where public data permits | How are vehicle markets and technology shares changing? |
| Mobility systems | Passenger and freight activity, road, rail, aviation, maritime, two-/three-wheelers | How are mobility demand and modal composition changing? |
| Energy and fuels | Electricity, oil products, gas, biofuels, hydrogen and synthetic fuels | How are transport energy sources and costs changing? |
| Charging and infrastructure | Public/private charging where measurable, power bands, access, grid interfaces | Where are infrastructure gaps and constraints emerging? |
| Battery and vehicle technology | Chemistry, cost, performance, manufacturing and recycling | Which technology trajectories can be supported by public evidence? |
| Industry and supply chains | Manufacturing, trade, critical minerals, components and industrial concentration | Where are dependencies and competitive shifts occurring? |
| Climate and environmental performance | Energy intensity, emissions factors, lifecycle boundaries and avoided emissions | What environmental outcomes follow under explicit assumptions? |
| Policy and macroeconomics | Regulations, incentives, prices, GDP, trade and energy security | Which external conditions shape transition pathways? |

The initial expansion should prioritize public, country-level, annual datasets. Subnational, model-level and company-level intelligence is conditional on reliable public or licensed data.

## 3. Shared intelligence core

Domain modules must use a common foundation without forcing their source data into one undifferentiated table.

The shared core should provide:

- Source registry and provenance.
- Versioned source metadata and retrieval/release dates.
- Explicit units, dimensions and definitions.
- Country/region identifiers and geography mappings.
- Temporal frequency and observation period.
- Coverage and missingness metadata.
- Validation rules and data-quality reports.
- Reproducible transformations.
- Model/experiment metadata and validation records.

A domain dataset retains its own schema. Integration occurs through documented domain contracts and derived analytical views, not by blindly joining unrelated measures.

## 4. Canonical observation contract

Every normalized observation should be traceable to the following conceptual fields, where applicable:

| Field | Meaning |
|---|---|
| `source_id` | Registered source identifier |
| `source_release` | Source edition or release |
| `source_retrieved_at` | Retrieval timestamp, if available |
| `geography_id` | Canonical country/region identifier |
| `geography_level` | Country, region, subnational, etc. |
| `period_start`, `period_end` | Observation period |
| `frequency` | Annual, quarterly, monthly, daily, etc. |
| `domain` | Automotive, mobility, energy, infrastructure, etc. |
| `measure` | Precisely defined indicator |
| `value` | Numeric observation |
| `unit` | Explicit unit |
| `technology`, `mode`, `fuel` | Applicable dimensions |
| `status` | Observed, estimated, modeled, projected |
| `coverage_status` | Available, missing, not reported, not applicable |
| `methodology_version` | Transformation/definition version |
| `quality_flags` | Validation and comparability warnings |

This is a target contract, not a claim that the current ETIL dataframes already contain these fields.

## 5. Integration principles

1. **Preserve domain semantics.** Vehicle stock, vehicle sales, passenger-kilometres, energy consumption and charging points are different measures and must not be conflated.
2. **Preserve native frequency.** Do not silently convert monthly fuel data, annual vehicle data and hourly electricity data into a common frequency.
3. **Preserve units and denominators.** Ratios must declare numerator, denominator, units and temporal alignment.
4. **Preserve source status.** Observed, estimated, modeled and projected values must remain distinguishable.
5. **Preserve missingness.** Missing data must not be interpreted as zero or as evidence of no activity.
6. **Version definitions.** Source revisions and methodology changes must be recorded rather than silently overwritten.
7. **Avoid false comparability.** Cross-country rankings require minimum coverage, comparable definitions and sensitivity analysis.
8. **Separate association from causality.** Cross-domain relationships are descriptive or predictive unless a valid causal design is established.
9. **No implicit future information.** Feature construction, imputation, scaling, winsorization, selection and calibration must respect the forecast origin.
10. **Keep integration optional.** Domain modules can be used independently; cross-domain analysis is an explicit analytical layer.

## 6. Cross-domain analytical products

The first cross-domain products should be narrow and interpretable:

### A. Mobility energy balance
Relate transport activity and vehicle/fuel mix to energy consumption, with explicit system boundaries and coverage.

### B. Technology transition comparison
Compare BEV, PHEV, HEV, ICE, hydrogen and relevant alternative fuels only where definitions and source coverage support comparison. Do not assume every technology is relevant to every transport mode.

### C. Energy cost and operating economics
Compare fuel and electricity costs using transparent consumption/efficiency assumptions. Label modeled TCO components separately from observed market prices.

### D. Infrastructure readiness
Connect vehicle/fleet indicators to charging or refueling infrastructure, preserving power class, access type, operational status and coverage limitations.

### E. Industrial and supply-chain exposure
Connect manufacturing, trade and critical-mineral indicators to technology and regional market structure, without implying company-level visibility where only national aggregates exist.

### F. Transition scenarios
Build scenarios from explicit assumptions about activity, technology shares, efficiency, energy prices, grid mix and policy. Scenario outputs are conditional explorations, not unconditional predictions.

## 7. Recommended implementation sequence

- **Foundation 0:** Temporal integrity, reproducible data contracts, source registry, and coverage/provenance reporting.
- **Foundation 1:** Generalize the canonical panel and data interfaces without breaking the current IEA EV pipeline.
- **Domain 1:** Add transport modes and non-EV powertrain/fuel comparisons using a separate domain adapter.
- **Domain 2:** Add transport energy balances and fuel/electricity economics.
- **Domain 3:** Add charging/refuelling infrastructure and geospatial layers where coverage permits.
- **Domain 4:** Add battery, industrial, trade and supply-chain research.
- **Integration:** Add cross-domain scenarios and composite indicators only after their component measures pass quality and validation gates.

## 8. Explicit exclusions for the initial expansion

Do not initially build:

- Real-time vehicle telemetry or sensor-fusion functionality.
- A fleet operations or route optimization product.
- Manufacturer-confidential cost or production models.
- A global all-energy data lake without defined research questions.
- A composite “transition score” that mixes incompatible domains or hides missing coverage.
- Causal claims from correlations or country-level aggregate data.
- A complex knowledge graph or distributed architecture before scale and use cases justify it.

## 9. Relationship to existing ETIL

The current IEA EV pipeline remains the first validated domain adapter. It should not be discarded or rewritten wholesale.

The existing diffusion, forecasting, regime and scoring modules should become reusable analytical capabilities only after their assumptions and temporal validation are explicit. Models must declare which domain and measure they support; a vehicle-adoption model cannot automatically be applied to energy demand or freight activity without a justified formulation.

## 10. Acceptance criteria for this architecture

Before new domains are implemented:

- Each domain has a documented research question and scope.
- Candidate sources, licenses, definitions, units and temporal coverage are assessed.
- A domain-specific data contract and quality checks are specified.
- Missingness and comparability rules are explicit.
- Cross-domain joins document keys, time alignment and assumptions.
- New models have baselines and temporal validation plans.
- Existing EV outputs remain reproducible and regression-tested.

**Principle:** Expand ETIL by adding well-defined research domains to a shared evidence and validation core—not by turning it into a collection of unrelated dashboards.
