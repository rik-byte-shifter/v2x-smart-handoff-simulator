from dataclasses import dataclass

# Paper / appendix: frozen seeds, dependency pins, and semantic ablation snapshots — ``config_reproducibility.py``.

@dataclass(frozen=True)
class SimConfig:
    # Performance
    headless_monte_carlo: bool = True

    # Channel modeling (interactive GUI defaults)
    enable_small_scale_fading: bool = False
    rician_k_factor: float = 10.0
    # Monte Carlo / batch eval: Rician (LOS) when unblocked, Rayleigh when blocked — robustness vs fast fading.
    monte_carlo_enable_small_scale_fading: bool = True

    # Monte Carlo defaults
    monte_carlo_trials: int = 100
    monte_carlo_duration_s: float = 30.0
    monte_carlo_base_seed: int = 42

    # Statistical reporting: bootstrap resamples for pairwise effect CIs in ``plotting_utils``.
    # Use 9999 for publication-grade stability; 1999 is a practical default for large factorial grids.
    bootstrap_n_resamples: int = 1999
    # ``compare_algorithms --quick`` uses this for ``run_statistical_tests`` / targeted bootstrap CIs.
    bootstrap_n_resamples_quick: int = 999

    # Factorial / batch runs: drop per-frame series from each trial dict after scalars are computed
    # (avoids MemoryError when ``all_results`` holds thousands of long trajectories).
    monte_carlo_store_trial_timeseries: bool = False
    # If True, ``MonteCarloEvaluator.run_monte_carlo`` appends each summary dict to ``results_history``
    # (can grow large during full factorial). Default False: history unused by comparison scripts.
    monte_carlo_record_evaluator_history: bool = False


CONFIG = SimConfig()
