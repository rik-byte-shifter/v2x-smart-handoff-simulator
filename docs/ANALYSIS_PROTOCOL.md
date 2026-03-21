# Pre-specified analysis protocol (paper / supplement)

This file records **what is primary vs exploratory** so results align with the code in `compare_algorithms.py` and `plotting_utils.py`.

## Primary outcome metric

- **External AoI (mean):** `avg_aoi_external` per trial; aggregate mean `avg_aoi_external_mean` in Monte Carlo summaries.
- Definition: independent packet process at fixed rate; success iff serving RSS > outage threshold (`compute_external_aoi` in `monte_carlo_runner.py`). Not the simulator’s internal AoI integrator.

## Pre-specified primary **inferential family** (FWER control)

- **Tests:** Proposed (`my_algorithm`) vs Robust A3+ (`robust_a3`) on **external AoI**, exactly the tuples in `compare_algorithms.PRIMARY_ENDPOINTS`.
- **Environment:** `urban_canyon`, weather `clear`, packet profile `critical` (see `PRIMARY_ENDPOINT_ENV`).
- **Trajectories:** `random_walk`, `linear`, `circular` (as listed in `PRIMARY_ENDPOINTS`).
- **Multiple comparison correction:** Holm–Bonferroni **only** across those tests; results in `outputs/primary_family_tests.csv`.

## Secondary metrics

- Handoff count, ping-pong count, outage time (serving RSS below threshold).
- Internal AoI (`avg_aoi`, `peak_aoi`): **exploratory** only (shared environment mechanics with some policies).

## Exploratory analyses (hypothesis-generating unless promoted)

- Full pairwise grid in `outputs/statistical_tests.csv` (all metrics including internal AoI).
- Subgroup tables pooling trajectories beyond the primary family.
- Any contrast not listed in `PRIMARY_ENDPOINTS`.

## Paper-facing statistical export subset

- `outputs/statistical_tests_primary_endpoints.csv`: rows where `metric` is external AoI, peak external AoI, handoffs, ping-pong, or outage (see `plotting_utils.PRIMARY_STATISTICAL_METRICS`).

## Effect sizes

- Continuous metrics: Cohen’s d (reported with mean difference and bootstrap CI).
- Discrete counts: **Cliff’s delta** (Cohen’s d omitted / NaN in CSV to avoid misleading huge values when variances are small).

## α level

- 0.05 for corrected primary family; exploratory contrasts reported without study-wide FWER claim unless otherwise noted.
