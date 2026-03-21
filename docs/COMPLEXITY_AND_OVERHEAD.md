# Computational complexity (decision-time sketch)

Rough order per simulation frame for the **proposed** policy (`update_handoff_logic`):

- **Measurements:** O(N towers) path-loss / shadowing checks; building LOS is O(N × B) segment tests (B = buildings); moving blockers add O(N × K) link–disk tests (K = blockers).
- **Lookahead / survival:** For each tower, short-horizon samples call `calculate_measurement` again → O(N × S) with S ≈ survival_window / survival_dt steps.
- **Baselines:** Greedy / A3-style are O(N); MPC is O(N × horizon_steps); velocity-aided uses a few lookahead RSS evaluations per tower per step.

The pygame loop targets **60 FPS** for visualization; headless Monte Carlo uses the same per-frame logic with `CONFIG.headless_monte_carlo`. Real RRC handover timelines are **orders of magnitude slower** than one frame; the simulator is a **policy algorithm testbed**, not a real-time RRC emulator. For a paper, add one sentence: decisions could be executed on a slower control loop (e.g. 10–100 ms) with subsampled kinematics if ported to a real stack.
