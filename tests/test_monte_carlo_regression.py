"""
Light regression tests: small n, short horizon, fixed seeds.
Ensures Monte Carlo harness and new baselines stay runnable headless.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from monte_carlo_runner import MonteCarloEvaluator, compute_external_aoi  # noqa: E402


@pytest.fixture
def evaluator() -> MonteCarloEvaluator:
    return MonteCarloEvaluator(base_scenario="urban_canyon")


def test_compute_external_aoi_monotone(evaluator: MonteCarloEvaluator) -> None:
    r = evaluator.run_single_trial(
        trial_id=0,
        duration_s=2.0,
        trajectory_pattern="linear",
        algorithm="my_algorithm",
        seed=999,
        scenario="urban_canyon",
        weather_mode="clear",
        packet_profile="critical",
        store_timeseries=True,
    )
    ext = compute_external_aoi(r)
    assert math.isfinite(ext)
    assert ext >= 0.0
    assert "avg_aoi_external" in r
    assert math.isfinite(r["avg_aoi_external"])


@pytest.mark.parametrize("algo", ["my_algorithm", "a3", "velocity_aided", "qlearning", "mpc"])
def test_baseline_runs_finite_external_aoi(evaluator: MonteCarloEvaluator, algo: str) -> None:
    r = evaluator.run_single_trial(
        trial_id=0,
        duration_s=2.0,
        trajectory_pattern="linear",
        algorithm=algo,
        seed=1001,
        scenario="urban_canyon",
        weather_mode="clear",
        packet_profile="critical",
    )
    assert math.isfinite(r["avg_aoi_external"])
    assert math.isfinite(r["handoff_count"])


def test_two_trial_regression_snapshot(evaluator: MonteCarloEvaluator) -> None:
    """Loose bounds so minor floating changes do not flake; catches gross breakage."""
    stats = evaluator.run_monte_carlo(
        n_trials=2,
        duration_s=3.0,
        trajectory_pattern="linear",
        algorithm="my_algorithm",
        base_seed=424242,
        scenario="urban_canyon",
        weather_mode="clear",
        packet_profile="critical",
    )
    m = stats["avg_aoi_external_mean"]
    assert 0.0 <= m < 5.0, f"unexpected external AoI mean {m}"
