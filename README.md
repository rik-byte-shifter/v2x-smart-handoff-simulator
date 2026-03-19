# Smart Handoff Simulator for V2X (Enterprise Edition)

A professional real-time simulator that demonstrates V2X/6G mobility behavior in a dynamic city environment, including predictive handoff and analytics tooling.

## What It Simulates

- Moving UE (mouse cursor) at 60 FPS
- Multiple static gNB/RSU towers
- Path loss and RSS in dBm
- Doppler shift from UE motion
- Building shadowing via LOS intersection
- Smart handoff with hysteresis + hold timer
- Predictive handoff (velocity and Kalman options)

## Enterprise Features

- **Scenario presets** (`1/2/3`): baseline, urban canyon, and intersection topologies
- **Telemetry export** (`X`): structured CSV with UE kinematics and per-tower metrics
- **Replay mode** (`R`): replays UE trajectory from the latest exported telemetry CSV
- **Performance charts** (`P`): auto-generates PNG analytics with RSS timeline and handoff timeline
- **Predictive model switch** (`K`): compare velocity extrapolation vs. Kalman-based forecasting
- **Professional dashboard**: live signal panel, connected tower state, mode indicators, and event log

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
- `X`: export telemetry CSV to `outputs/`
- `P`: generate performance chart PNG to `outputs/`
- `R`: replay UE trajectory from latest exported CSV
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
- Handoff trigger:
  - Current strongest-cell margin check
  - Predictive near-future risk check
  - Minimum hold time to avoid ping-pong
- Kalman forecaster:
  - Constant-velocity alpha-beta style state update
  - Used for second predictive handoff model
- Telemetry and chart outputs:
  - CSV: `outputs/telemetry_YYYYMMDD_HHMMSS.csv`
  - PNG: `outputs/performance_YYYYMMDD_HHMMSS.png`

## Why This Is Unique

Unlike simple strongest-signal demos, this simulator combines:

- Real-time wireless metrics
- Geometry-based signal obstruction
- Two predictive models (including Kalman forecasting)
- Deterministic replay for demo reproducibility
- Analytics export pipeline for portfolio-ready results
