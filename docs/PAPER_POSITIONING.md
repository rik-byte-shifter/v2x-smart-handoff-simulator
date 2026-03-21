# Paper positioning and reviewer-facing language

## Simulation scope

This repository supports a **controlled geometric simulation study** of handoff policies. Claims should **not** imply calibrated 3GPP field performance or a validated digital twin.

## Vocabulary

| Avoid alone | Prefer |
|-------------|--------|
| “Semantic handoff” (information-theoretic connotation) | **Urgency-aware composite utility** or **AoI- and load-aware handoff score** |
| “Survival score” (statistical survival analysis) | **Short-horizon connectivity survival heuristic** or **risk-aware lookahead score** |
| “Semantic weights” in prose | **Utility weights** (code identifiers `SEMANTIC_W_*` are legacy names) |

## Trajectory caveat: `random_walk`

Highly erratic synthetic motion can favor reactive strongest-cell policies under some cells; **pre-specified primary inference** uses a fixed set of trajectories and environment (`PRIMARY_ENDPOINTS`). If external AoI is flat or negative vs A3 on `random_walk` in some cells, report it as a **boundary condition** or **regime limitation**, not as a hidden failure—see factorial outputs and discussion text.

## Fading and GUI vs Monte Carlo

- **Interactive GUI default:** small-scale fading **off** (`config.CONFIG.enable_small_scale_fading`).
- **Default batch / Monte Carlo:** fading **on** (`CONFIG.monte_carlo_enable_small_scale_fading = True`) unless you change `config.py`.

State this explicitly in Methods so reviewers do not see a contradiction.

## Future work (high impact, not in this repo)

- **ns-3 / Simu5G** or another standard packet simulator with the same policy API.
- **Trace-driven mobility** (SUMO, open vehicular traces).
- **Formal models:** AoI with switching costs in a simplified Markov/renewal framework (structural results or bounds)—a separate theory contribution.

## Weight tuning and `semantic_weight_tuner.py`

- Default weights live in `src/main.py` (`SEMANTIC_W_*`).
- The optional genetic tuner searches weights against a **composite fitness** on the same simulator—treat as **hyperparameter search**, not independent validation.
- For publication: report defaults used in main experiments; if tuner outputs are used, disclose tuning protocol and use a **held-out** scenario seed set or nested evaluation to reduce overfitting to the simulator (tuner does not do this automatically).
