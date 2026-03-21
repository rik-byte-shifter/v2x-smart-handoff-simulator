"""
Frozen snapshot for paper reproducibility (software, seeds, key constants, ablations).

These values are **documentation**: the running simulator still reads live constants from
``src/main.py`` and ``config.py``. When publishing, update ``python_version`` from
``python --version`` and pin ``dependencies`` to match ``pip freeze`` if reviewers require it.

Semantic weights match ``SEMANTIC_W_*`` in ``main.py`` (non-negative; the score formula
subtracts load and AoI terms).
"""

from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Master bundle (appendix / supplementary material)
# ---------------------------------------------------------------------------
REPRODUCIBILITY_CONFIG: Dict[str, Any] = {
    "software": {
        # Update at submission from: python --version
        "python_version": "3.13.7",
        "dependencies": {
            "pygame": ">=2.5.0",
            "numpy": ">=1.24.0",
            "scipy": ">=1.11.0",
            "matplotlib": ">=3.7.0",
            "seaborn": ">=0.12.0",
            "pandas": ">=2.0.0",
        },
    },
    "random_seeds": {
        "monte_carlo_base_seed": 42,
        "seed_increment_per_trial": 1,
        "bootstrap_mean_diff_random_state": 0,
    },
    "simulation_constants_paper": {
        "METER_PER_PIXEL": 0.1,
        "FREQ_HZ": 5.9e9,
        "PATH_LOSS_EXPONENT": 2.4,
        "REFERENCE_DISTANCE_M": 1.0,
        "REFERENCE_LOSS_DB": 32.44,
        "TX_POWER_DBM": 30.0,
        "SHADOWING_DROP_DB": 20.0,
        "OUTAGE_RSS_THRESHOLD_DBM": -95.0,
        "HANDOFF_MARGIN_DB": 5.0,
        "HANDOFF_HOLD_S": 0.5,
        "HANDOFF_SPIKE_S": 0.035,
        "FPS": 60,
        "external_aoi_packet_rate_hz_default": 10.0,
    },
    "config_py_defaults": {
        "headless_monte_carlo": True,
        "enable_small_scale_fading": False,
        "rician_k_factor": 10.0,
        "monte_carlo_enable_small_scale_fading": True,
        # Default trial count for standalone ``monte_carlo_runner.py`` / ``config.CONFIG`` runs.
        # Factorial batch evaluation in ``compare_algorithms.py`` uses ``PRIMARY_TRIALS`` (100)
        # and ``SECONDARY_TRIALS`` (25) per cell instead—see that module for power allocation.
        "monte_carlo_trials": 100,
        "monte_carlo_duration_s": 30.0,
        "monte_carlo_base_seed": 42,
        "bootstrap_n_resamples": 1999,
    },
    # Matches monte_carlo_runner.MonteCarloEvaluator._ablation_profile + full policy.
    "semantic_weights": {
        "full": {
            "survival": 1.0,
            "predicted_rss": 0.22,
            "load": 0.60,
            "aoi": 0.80,
        },
        "no_survival": {
            "survival": 0.0,
            "predicted_rss": 0.22,
            "load": 0.60,
            "aoi": 0.80,
        },
        "no_aoi": {
            "survival": 1.0,
            "predicted_rss": 0.22,
            "load": 0.60,
            "aoi": 0.0,
        },
        "no_load": {
            "survival": 1.0,
            "predicted_rss": 0.22,
            "load": 0.0,
            "aoi": 0.80,
        },
    },
    "ablation_boolean_flags": {
        "full": {
            "enable_predictive_trigger": True,
            "enable_relay_fallback": True,
        },
        "reactive_only": {
            "enable_predictive_trigger": False,
            "enable_relay_fallback": True,
        },
        "no_relay": {
            "enable_predictive_trigger": True,
            "enable_relay_fallback": False,
        },
    },
    "notes": [
        "Ablation runs apply merged knob dicts per trial on each V2XSmartHandoffSimulator instance (see MonteCarloEvaluator._merged_policy_profile).",
        "compare_algorithms factorial: PRIMARY_TRIALS / SECONDARY_TRIALS in compare_algorithms.py "
        "(urban_canyon × trajectories use primary; optional uniform n_trials= overrides all).",
        "Tower load = scheduled component + cross-traffic from per-tower invisible UE sessions (birth–death); see BACKGROUND_* in src/main.py.",
        "Monte Carlo uses CONFIG.monte_carlo_enable_small_scale_fading (Rician LOS / Rayleigh NLOS) via sim.enable_small_scale_fading_override; GUI uses CONFIG.enable_small_scale_fading unless override is set.",
    ],
}


def get_reproducibility_config() -> Dict[str, Any]:
    """Return a copy of the frozen reproducibility bundle (safe to mutate)."""
    import copy

    return copy.deepcopy(REPRODUCIBILITY_CONFIG)
