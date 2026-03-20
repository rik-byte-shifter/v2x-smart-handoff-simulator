from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    # Performance
    headless_monte_carlo: bool = True

    # Channel modeling
    enable_small_scale_fading: bool = False
    rician_k_factor: float = 10.0

    # Monte Carlo defaults
    monte_carlo_trials: int = 100
    monte_carlo_duration_s: float = 30.0
    monte_carlo_base_seed: int = 42


CONFIG = SimConfig()
