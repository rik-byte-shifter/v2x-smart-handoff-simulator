# Architecture and Signal Logic

## 1) Coordinate System (World Layer)

- Screen is a 2D city map.
- UE coordinate = current mouse position.
- Towers are fixed `(x, y)` coordinates.
- Buildings are rectangles drawn by user drag action.
- Conversion scale: `1 px = 0.1 m`.

## 2) Physics Engine (Signal Layer)

### Distance

`d = sqrt((x_ue - x_tower)^2 + (y_ue - y_tower)^2)`

Then convert to meters with the configured scale.

### Path Loss / RSS

The project uses a log-distance approximation:

`PL(d) = PL(d0) + 10 * n * log10(d / d0)`

`RSS(dBm) = TX_POWER - PL(d)`

### Shadowing by Buildings

- For each tower-UE pair, check if the LOS segment intersects any building rectangle.
- If yes, apply attenuation:

`RSS = RSS - 20 dB`

### Doppler

Using radial velocity from UE motion and tower direction:

`f_d = v_r / lambda`

where `lambda = c / f`.

## 3) Smart Handoff Brain (Decision Layer)

### Baseline candidate

- Find strongest currently measured tower by RSS.

### Predictive candidate pressure

- Forecast UE position over a short horizon.
- Recompute future RSS for each tower.
- If current tower is predicted to degrade relative to alternatives, allow a proactive switch.

Two forecasting modes are available:

- `velocity` mode: direct extrapolation from measured UE velocity.
- `kalman` mode: alpha-beta style constant-velocity filtering and prediction.

### Hysteresis + Hold Time

- Candidate tower must exceed margin (`5 dB`) or satisfy predictive pressure.
- Candidate must persist for hold time (`0.5s`) before final handoff.
- This reduces ping-pong oscillation.

## 4) Eventing and UX

- Every major action is timestamped:
  - simulation start
  - building creation
  - normal handoff
  - predictive handoff

This makes the simulator useful for demos, class presentations, and algorithm explanation.

## 5) Enterprise Tooling

### Scenario Presets

- Keyboard presets instantly apply curated city layouts:
  - baseline
  - urban canyon
  - intersection

### Dynamic Safe-Placement System

- Towers are placed through a safety filter that rejects points colliding with UI rectangles.
- Protected regions currently include:
  - right dashboard panel
  - top-left help/controls panel
- On scenario switch (or help panel toggle), tower coordinates are re-evaluated and snapped to the nearest valid point.

### Telemetry Pipeline

- Every frame is logged in-memory with:
  - UE position and velocity
  - connected and candidate tower
  - model mode and replay state
  - per-tower RSS, Doppler, distance, and blockage flags
- CSV export writes to `outputs/telemetry_*.csv`.

### Replay Engine

- Replay mode loads UE trajectory from exported telemetry and drives the UE automatically.
- This enables deterministic demos for interview/portfolio presentations.

### Performance Charts

- Matplotlib output (`outputs/performance_*.png`) includes:
  - RSS-over-time curves for all towers
  - connected-tower timeline (handoff sequence)

### Side-by-Side A/B Evaluator

- A/B mode runs both predictive policies on the same trajectory:
  - A: velocity extrapolation
  - B: Kalman-based forecast
- It exports summary metrics to `outputs/ab_eval_*.csv`, including:
  - handoff count
  - ping-pong count
  - average connected RSS
  - outage time below threshold
- It also generates `outputs/ab_eval_*.png` for visual comparison.
