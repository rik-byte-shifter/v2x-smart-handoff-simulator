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

### Dynamic Stochastic Shadowing (Moving Blockers)

- The simulator also models moving blockers (pedestrian/bus/truck) as mobile circles with independent velocity vectors.
- Each blocker class has a different attenuation profile and time-varying stochastic jitter.
- If a tower-UE link segment intersects a moving blocker footprint, additional loss is injected:

`RSS = RSS - dynamic_drop(kind, time, jitter)`

### Ghost Blocker (Sensor Uncertainty)

- For future-state evaluation, blocker position is perturbed with stochastic sensor noise.
- This prevents unrealistically perfect foresight and makes the policy robust to noisy measurements.
- **Scope:** This is **forward prediction inside the same geometry + path-loss model**—not a calibrated *digital twin* of a real deployment (no fused maps, no live RAN telemetry, no external validation).

### Doppler

Using radial velocity from UE motion and tower direction:

`f_d = v_r / lambda`

where `lambda = c / f`.

## 3) Smart Handoff Brain (Decision Layer)

### Baseline candidate

- Find strongest currently measured tower by RSS.

### Robust A3+ (stronger baseline)

- Same hysteresis and time-to-trigger (TTT) as the simplified A3 event, but **candidates are ranked by load-adjusted RSS**: `RSS − k · load` using simulator `tower_loads`. A handoff is considered toward the best alternative when its effective RSS exceeds **raw** serving RSS by the hysteresis margin, and the condition persists for TTT.

### Predictive candidate pressure

- Forecast UE position over a short horizon.
- Recompute future RSS for each tower, including predicted positions of moving blockers.
- If current tower is predicted to degrade relative to alternatives, allow a proactive switch.

Two forecasting modes are available:

- `velocity` mode: direct extrapolation from measured UE velocity.
- `kalman` mode: alpha-beta style constant-velocity filtering and prediction.

### Risk-Aware Lookahead (“Survival” Heuristic)

- A **connectivity survival heuristic** is evaluated per tower over a future window (not Cox/Kaplan–Meier survival analysis).
- At each step in the window, outage probability is estimated from predicted RSS.
- The algorithm prefers the tower with the highest heuristic score (with RSS as a tie-strength term).
- This can intentionally pick a currently weaker tower when it is predicted to remain stable while a stronger one is about to be blocked.

### Urgency-Aware Composite Handoff (implementation: “semantic” score in code)

- Paper wording: **urgency-aware composite utility** (not information-theoretic semantic communication).
- Data urgency profile: `critical` (strict freshness target) vs `standard` (relaxed).
- For each candidate tower, the policy estimates:
  - predicted RSS / connectivity-survival heuristic
  - predicted tower congestion (load)—**synthetic** load trace in the simulator, not measured congestion
  - projected Age of Information ratio
  - handoff latency spike cost
- Final tower choice is a weighted composite score, not RSS alone.

### Cooperative V2V Relay Fallback

- If tower connectivity is predicted to collapse, the UE can temporarily relay traffic via a nearby V2V node.
- This creates a hybrid mode-switch (`tower` <-> `relay`) instead of only tower-to-tower switching.

### Environmental Entropy Layer

- Weather mode (`clear/rain/fog/storm`) changes uncertainty intensity and handoff conservatism.
- Higher entropy increases stochastic shadow jitter and effectively raises the margin needed for aggressive switching.

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

## 5) Tooling and scenarios

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
  - per-tower survival score (risk-aware signal continuity metric)
- CSV export writes to `outputs/telemetry_*.csv`.

### Replay Engine

- Replay mode loads UE trajectory from exported telemetry and drives the UE automatically.
- This enables deterministic replays of a fixed trajectory for testing and figures.

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

## 6) Simulation scope and limitations (claims)

This stack is a **geometric 2D simulator**: log-distance path loss, binary building shadowing, heuristic moving-blocker attenuation, and Doppler from UE motion. It **does not** implement full **3GPP TR 38.901** channel models, interference-limited scheduling, or **map- / trace-driven** mobility. Outcomes should be read as **algorithmic relative performance** under controlled conditions, not absolute field performance.

**Parameters** (see `src/main.py`, `config.py`): η = 2.4, PL(d₀) = 32.44 dB at d₀ = 1 m and f = 5.9 GHz, −20 dB building LOS penalty, TX 30 dBm, optional Rician/Rayleigh fading (**off by default** in `config.py`).

Longer narrative, blocker-class table, and repo cross-checks: **`docs/SIMULATION_FIDELITY_AND_LIMITATIONS.md`**.
