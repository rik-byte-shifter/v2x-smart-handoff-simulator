# V2X Smart Handoff Simulator (Research Edition)

A simulation framework for studying proactive handoff algorithms in V2X networks with Age-of-Information optimization. It includes an interactive 2D map simulator and batch Monte Carlo tools for reproducible comparisons.

## Research Contributions

- Predictive handoff with a **short-horizon connectivity survival heuristic** (RSS-based lookahead; not parametric survival analysis)
- **Urgency-aware composite utility** (predicted RSS, synthetic load, projected AoI ratio); see `docs/PAPER_POSITIONING.md` for reviewer-facing wording
- V2V relay fallback for outage resilience

## Citation

If you use this code in your research, please cite:

> [Your paper title, venue, year]

## Limitations

This is a **simulation study** using a **2D geometric channel model**. Results show **algorithmic relative performance** under controlled assumptions, not calibrated **absolute field performance** or 3GPP-standard channel fidelity. See `docs/SIMULATION_FIDELITY_AND_LIMITATIONS.md` for scope, assumptions, and what is not claimed.

## What It Simulates

- Moving UE (mouse cursor) at 60 FPS
- Multiple static gNB/RSU towers
- Path loss and RSS in dBm
- Doppler shift from UE motion
- Building shadowing via LOS intersection
- Dynamic stochastic shadowing from moving blockers (pedestrian/bus/truck classes)
- Smart handoff with hysteresis + hold timer
- Predictive handoff (velocity and Kalman options)
- Risk-aware survival handoff using **predicted UE motion**, **predicted blocker positions with additive Gaussian noise** (noisy future-state geometry), and **near-future RSS**—not a calibrated network digital twin
- Urgency-aware handoff (`critical` vs `standard` packet profiles; composite score, not information-theoretic semantic communication)
- Predicted tower congestion-aware decision scoring
- Handoff latency spike cost modeling (switches are no longer "free")
- V2V relay fallback path for near-outage moments
- Weather entropy modes that increase channel uncertainty and conservatism

## Features

- **Scenario presets** (`1/2/3`): baseline, urban canyon, and intersection topologies
- **Dynamic safe-placement**: towers are auto-positioned away from UI overlays (dashboard/help)
- **Telemetry export** (`X`): structured CSV with UE kinematics and per-tower metrics
- **Replay mode** (`R`): replays UE trajectory from the latest exported telemetry CSV
- **Performance charts** (`P`): auto-generates PNG analytics with RSS timeline and handoff timeline
- **Predictive model switch** (`K`): compare velocity extrapolation vs. Kalman-based forecasting
- **Side-by-side A/B evaluator** (`E`): compares velocity vs Kalman predictors and exports metrics/charts
- **In-app dashboard**: live signal panel, connected tower state, mode indicators, and event log
- **Moving blockers**: obstacle classes with stochastic attenuation; proposed policy can rank candidates with a survival-style term
- **Packet urgency modes** (`U`): `critical` vs `standard` deadlines; handoff scoring includes load and projected AoI ratio (synthetic load)

## Architecture

The simulator contains three layers:

1. **World layer**: 2D coordinate system, towers, UE position, buildings
2. **Physics layer**: path loss + Doppler + LOS shadowing computations
3. **Decision layer**: strongest-cell evaluation, predictive scoring, hysteresis/hold-time handoff

See `docs/ARCHITECTURE.md` for details.

**Paper / evaluation:** Pre-specified **primary endpoint = external AoI** (independent RSS-threshold packet process); secondary = handoffs, ping-pong, outage, etc. See **`docs/PRIMARY_AND_SECONDARY_ENDPOINTS.md`**, **`docs/ANALYSIS_PROTOCOL.md`** (primary vs exploratory), and **`docs/PAPER_POSITIONING.md`** (terminology, fading, `random_walk` caveat).

## Controls

- Move mouse: move UE
- Left-click + drag: draw building
- `1`: load baseline scenario
- `2`: load urban canyon scenario
- `3`: load intersection scenario
- `K`: switch predictive model (`velocity` / `kalman`)
- `U`: toggle packet urgency (`critical` / `standard`)
- `W`: cycle weather entropy (`clear` / `rain` / `fog` / `storm`)
- `X`: export telemetry CSV to `outputs/`
- `P`: generate performance chart PNG to `outputs/`
- `R`: replay UE trajectory from latest exported CSV
- `E`: run A/B evaluator (velocity vs kalman)
- `C`: clear all buildings
- `H`: toggle help
- `Esc`: exit

## Reproducibility (paper)

Frozen **software pins, seeds, channel/handoff constants, and ablation weight snapshots** for an appendix or supplementary table: **`config_reproducibility.py`** (`REPRODUCIBILITY_CONFIG`). For exact dependency versions from a machine that ran the experiments, use **`requirements-lock.txt`** (`pip freeze`). Update the recorded Python version with `python --version` before submission.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Tests (optional):** `pip install -r requirements-dev.txt` then `python -m pytest tests/`.

## Run

```bash
python src/main.py
```

## Factorial evaluation (`compare_algorithms.py`)

**Runtime:** The default **full** factorial is **120 cells** (5 trajectories × 3 scenarios × 4 weathers × 2 profiles) × **7 algorithms** × up to **100 trials** × **30 s** simulated time × 60 FPS — expect **many hours**. For development, use:

```bash
python compare_algorithms.py --quick
```

**Quick mode** runs only **3** trajectories × **1** environment (`urban_canyon`, `clear`, `critical`) × **20** trials × **15 s** sim (~100×+ fewer frame-equivalents than the full grid). Optional: `--slim-baselines` (drops greedy + Q-learning), `--n-trials 8`, `--duration-s 10`.

`python compare_algorithms.py` (no flags) runs the **full factorial** over map scenarios (`baseline`, `urban_canyon`, `intersection`), **weather** (`clear`, `rain`, `fog`, `storm`), **packet profile** (`critical`, `standard`), and **trajectory** patterns in `compare_algorithms.TRAJECTORY_PATTERNS`. **Pre-specified primary inference** uses a subset of trajectories; erratic patterns such as `random_walk` may interact differently with policies—discuss in paper if contrasts differ (see `docs/PAPER_POSITIONING.md`). **Urban canyon** cells use **`PRIMARY_TRIALS`** (100) per algorithm; **baseline** and **intersection** use **`SECONDARY_TRIALS`** (25). Pass `n_trials=` to `run_comprehensive_comparison()` to force one count for every cell.

Outputs include **`outputs/subgroup_proposed_aoi.csv`** (per-trajectory proposed metrics; **external AoI** columns are primary), **`outputs/subgroup_analysis_table.csv`** (**primary:** Proposed vs A3 **external** AoI mean over trajectories + **Improvement_External**; **exploratory:** internal AoI columns), **`outputs/statistical_tests.csv`** (all metrics: external + internal AoI, handoffs, etc.; **Cohen’s d omitted for discrete counts**, **Cliff’s delta** reported), **`outputs/statistical_tests_primary_endpoints.csv`** (same grid, **paper-facing subset**: external AoI + churn + outage only), **`outputs/targeted_proposed_vs_a3_tests.csv`** (Proposed vs A3; metrics include external AoI first), **`outputs/primary_family_tests.csv`** (pre-specified **primary inferential family**: external AoI, Proposed vs Robust A3+, three trajectories, **Holm** only across that set), **`outputs/comparison_factorial/…`** (CDFs/boxplots including **avg_aoi_external**), and summary boxplots in **`outputs/comparison_all_metrics/`** filtered to **urban_canyon / clear / critical** by default.

**Sensitivity (appendix):** `python sensitivity_sweep.py` → `outputs/sensitivity_sweep_summary.csv` (one-at-a-time channel / utility knobs).

**Ablation figures (appendix):**
```bash
python compare_algorithms.py --ablation
```
Defaults: 50 trials × 30 s × five trajectories (long). Faster appendix pass:
```bash
python compare_algorithms.py --ablation --ablation-trials 20 --ablation-duration 15 --ablation-patterns random_walk,linear,circular
```
Writes `outputs/ablation_<pattern>/` (CDFs + boxplots), `outputs/statistical_tests_ablation*.csv`, `outputs/ablation_all_metrics/`.

Pass `full_factorial=False` to `run_comprehensive_comparison()` for a quick three-pattern run with default simulator weather/profile (still uses primary trial count on urban_canyon). Pass `n_trials=25` for a uniform low count everywhere.

## Technical Notes

- **Robust A3+ baseline** (`robust_a3` / `a3+`): same TTT/hysteresis structure as A3, but candidate ranking uses **load-adjusted RSS** (`RSS − load_offset × load`) vs raw serving RSS; stronger than plain greedy RSS.
- **MPC lookahead baseline** (`mpc` / `mpc_lookahead`): sums predicted RSS over a short horizon (**default 0.8 s** aligned with `PREDICT_LOOKAHEAD_S`, step 0.1 s) using `get_predicted_position` and `calculate_measurement` with `future_s`; subtracts a one-time **handoff cost** (default **5 dB**) at step 0 when changing cells.
- **Velocity-aided RSS baseline** (`velocity_aided`): lookahead RSS ranking + hysteresis/TTT (literature-style mobility hint without the full proposed composite score).
- **GUI vs Monte Carlo fading:** interactive runs default to **no** small-scale fading (`config.enable_small_scale_fading`); headless Monte Carlo defaults to **Rician/Rayleigh on** (`monte_carlo_enable_small_scale_fading`). State both in paper methods—see `docs/PAPER_POSITIONING.md`.
- Frequency: 5.9 GHz (typical ITS band usage context)
- RSS model: log-distance path loss approximation
- Shadowing: fixed 20 dB attenuation when LOS intersects a building rectangle
- Dynamic shadowing: moving blockers add class-dependent stochastic attenuation on the link
- Ghost blocker uncertainty: predicted blocker positions include sensor-noise offset
- Handoff trigger:
  - Current strongest-cell margin check
  - Predictive near-future risk check
  - Survival score pressure over a future time window (risk-aware proactive switch)
  - AoI/latency/load-aware semantic pressure (urgency vs quality policy)
  - Minimum hold time to avoid ping-pong
- Kalman forecaster:
  - Constant-velocity alpha-beta style state update
  - Used for second predictive handoff model
- Telemetry and chart outputs:
  - CSV: `outputs/telemetry_YYYYMMDD_HHMMSS.csv`
  - PNG: `outputs/performance_YYYYMMDD_HHMMSS.png`
- A/B evaluator outputs:
  - CSV: `outputs/ab_eval_YYYYMMDD_HHMMSS.csv`
  - PNG: `outputs/ab_eval_YYYYMMDD_HHMMSS.png`

## Model scope (estimation vs. validation)

**Forward-modeled inside the simulator** (same path-loss geometry rolled forward): UE pose (velocity or Kalman-style filter), blocker pose with Gaussian perturbation (sensing-error stand-in, not a learned stack), short-horizon RSS/outage risk, heuristic survival pressure, and **synthetic** tower load (proxy only).

**Validated here:** in-simulator Monte Carlo and scripted trajectories—**internal consistency** with the above, not field truth.

**Not claimed:** trace-driven or drive-test validation, 3GPP-calibrated channels, or production digital-twin fidelity. See **`docs/SIMULATION_FIDELITY_AND_LIMITATIONS.md`** and **`docs/ARCHITECTURE.md`** §6.

## Compared to strongest-RSS-only demos

This codebase adds:

- Per-frame wireless metrics in a 2D map
- Geometry-based obstruction (buildings + moving blockers)
- Forward-looking handoff pressure from **perturbed future-state geometry** (noisy predicted blocker positions), using the same path-loss model as the current step
- A proposed policy that can prefer a **weaker-but-more-stable** candidate (survival-style term) and semantic/AoI-aware scoring
- Two UE motion predictors for the lookahead path (velocity vs Kalman-style filter)
- Replay from exported trajectories and CSV/PNG export for evaluation scripts (`compare_algorithms.py`, `monte_carlo_runner.py`)
