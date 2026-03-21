# Pre-specified primary and secondary endpoints (paper text)

Use this block (adapt section numbers) in your **Methods / Evaluation** or **Experimental Design** section so reviewers see a **single primary outcome** and no “fishing” across metrics.

---

## Markdown version

**Primary and secondary outcomes**

We **pre-specify *external Age of Information (AoI)*** as our **primary endpoint**. External AoI is the time-averaged instantaneous age, under an **independent packet process** that does not use the handoff policy’s semantic weights or internal AoI integrator: packets arrive at a fixed rate; each scheduled reception succeeds if and only if the **serving RSS** exceeds a fixed outage threshold (same physical layer geometry as the simulator). Lower external AoI is better. Implementation: `compute_external_aoi()` in `monte_carlo_runner.py` (trial-level mean over the Monte Carlo window; default packet rate 10 Hz unless stated).

**Note on primary family:** External AoI is the primary **metric** (the outcome we emphasize in conclusions). That is not the same thing as the pre-specified primary **inference family** in code: `compare_algorithms.PRIMARY_ENDPOINTS` fixes a *small set of hypothesis tests* on that metric (e.g. external AoI, Proposed vs Robust A3+, three trajectories, one environment cell). **Holm–Bonferroni** in `outputs/primary_family_tests.csv` applies **only across that family**. Other contrasts—including the full pairwise grid in `statistical_tests.csv` and the separate **Proposed vs A3** targeted table—are **secondary or exploratory** unless you explicitly add them to `PRIMARY_ENDPOINTS`.

**Secondary endpoints** (reported for context, not used to change the primary conclusion without disclosure):

- **Handoff count** and **ping-pong count** (network churn / stability).
- **Outage time** (fraction of time below RSS threshold on the serving link).
- **Internal (simulator) AoI** — the integrated AoI state used inside the environment; we report it as **exploratory** because it shares mechanisms with some policies and is not our primary inferential target.

**Exploratory analyses** may include peak AoI, average RSS, or subgroup tables (e.g. Proposed vs A3 by scenario); these are **hypothesis-generating** unless pre-specified in a protocol addendum.

---

## Short version (plain paragraph)

We pre-specify **external AoI** as our **primary outcome metric**: the time average of instantaneous age under an independent packet process (fixed packet rate; reception succeeds if and only if serving RSS exceeds a fixed outage threshold). That process does not use the handoff policy’s semantic weights or the simulator’s internal AoI integrator. **Secondary metrics** include handoff count, ping-pong count, and outage time. We report **internal AoI** only as exploratory, since it is coupled to the same environment dynamics as some policies. Primary conclusions should be framed on **external AoI** unless an analysis is explicitly labelled exploratory.

---

## Alignment with code

| Endpoint            | Code / output |
|---------------------|----------------|
| Primary: external AoI (mean) | `compute_external_aoi()`, trial field `avg_aoi_external`, Monte Carlo `avg_aoi_external_mean` |
| Primary (peak, optional)     | `peak_aoi_external` |
| Secondary: handoffs / ping-pong | `handoff_count`, `ping_pong_count` per trial |
| Secondary: outage            | `outage_time_s` |
| Exploratory: internal AoI    | `avg_aoi`, `peak_aoi` |

**Targeted inference (Proposed vs A3):** `plotting_utils.test_proposed_vs_a3` and `run_targeted_proposed_vs_a3_all_cells` produce `outputs/targeted_proposed_vs_a3_tests.csv`—trial-level tests **pooled across trajectory patterns** per (scenario, weather, profile), aligned with the subgroup improvement table (not the all-pairwise `statistical_tests.csv`).

**Primary family (FWER across a pre-specified set):** `compare_algorithms.run_primary_tests` and `PRIMARY_ENDPOINTS` define a small family (e.g. external AoI, Proposed vs Robust A3+, three trajectories, fixed weather/profile matching the default plot filter); Holm correction applies **only across that family**, with results in `outputs/primary_family_tests.csv`. The full `statistical_tests.csv` grid is **exploratory** outside this family unless you expand `PRIMARY_ENDPOINTS`.

---

## Why this satisfies reviewers

- **One primary endpoint** reduces the appearance of p-hacking across many metrics.
- **External AoI** is defined without the policy’s semantic objective, improving separation from the proposed algorithm’s internal loss.
- No new experiments are required—only **clear writing** and **consistent emphasis** in abstract, conclusion, and figures (e.g. lead with external AoI in the main comparison figure).
