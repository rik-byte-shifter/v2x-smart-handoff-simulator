# Smart Handoff Simulator for V2X (Enterprise Edition)

A professional real-time simulator that demonstrates V2X/6G mobility behavior in a dynamic city environment, including predictive handoff and analytics tooling.

## What It Simulates

- Moving UE (mouse cursor) at 60 FPS
- Multiple static gNB/RSU towers
- Path loss and RSS in dBm
- Doppler shift from UE motion
- Building shadowing via LOS intersection
- Dynamic stochastic shadowing from moving blockers (pedestrian/bus/truck classes)
- Smart handoff with hysteresis + hold timer
- Predictive handoff (velocity and Kalman options)
- Risk-aware survival handoff using digital-twin future blockage checks
- AoI-driven semantic handoff (`critical` vs `standard` packet urgency)
- Predicted tower congestion-aware decision scoring
- Handoff latency spike cost modeling (switches are no longer "free")
- V2V relay fallback path for near-outage moments
- Weather entropy modes that increase channel uncertainty and conservatism

## Enterprise Features

- **Scenario presets** (`1/2/3`): baseline, urban canyon, and intersection topologies
- **Dynamic safe-placement**: towers are auto-positioned away from UI overlays (dashboard/help)
- **Telemetry export** (`X`): structured CSV with UE kinematics and per-tower metrics
- **Replay mode** (`R`): replays UE trajectory from the latest exported telemetry CSV
- **Performance charts** (`P`): auto-generates PNG analytics with RSS timeline and handoff timeline
- **Predictive model switch** (`K`): compare velocity extrapolation vs. Kalman-based forecasting
- **Side-by-side A/B evaluator** (`E`): compares velocity vs Kalman predictors and exports metrics/charts
- **Professional dashboard**: live signal panel, connected tower state, mode indicators, and event log
- **Never-done-before mobility model**: moving obstacle classes with stochastic attenuation + survival-first tower decisions
- **AoI-driven cognitive mode-switching**: balances urgency vs quality under tower load and weather entropy

## Architecture

The simulator contains three layers:

1. **World layer**: 2D coordinate system, towers, UE position, buildings
2. **Physics layer**: path loss + Doppler + LOS shadowing computations
3. **Decision layer**: strongest-cell evaluation, predictive scoring, hysteresis/hold-time handoff

See `docs/ARCHITECTURE.md` for details.

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

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

## Technical Notes

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

## Why This Is Unique

Unlike simple strongest-signal demos, this simulator combines:

- Real-time wireless metrics
- Geometry-based signal obstruction
- Moving-object stochastic obstruction (not only static map blockage)
- Digital-twin-style future obstacle prediction in handoff decisions
- Survival-first policy that may intentionally choose a weaker-but-safer tower
- Two predictive models (including Kalman forecasting)
- Deterministic replay for demo reproducibility
- Analytics export pipeline for portfolio-ready results
