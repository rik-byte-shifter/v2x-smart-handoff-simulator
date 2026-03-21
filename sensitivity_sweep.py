"""
One-at-a-time sensitivity of the **primary metric** (external AoI) to selected channel
and policy hyperparameters. Intended for appendix figures / robustness paragraphs.

Usage:
  python sensitivity_sweep.py
  python sensitivity_sweep.py --trials 12 --pattern linear

Restores ``src.main`` globals after each sweep arm.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import monte_carlo_runner as mcr  # noqa: E402
from monte_carlo_runner import MonteCarloEvaluator  # noqa: E402

import main as sim_main  # noqa: E402


def _run_cell(
    label: str,
    apply_patch: Callable[[], None],
    restore: Callable[[], None],
    n_trials: int,
    pattern: str,
    base_seed: int,
) -> Dict[str, Any]:
    apply_patch()
    try:
        ev = MonteCarloEvaluator(base_scenario="urban_canyon")
        prop = ev.run_monte_carlo(
            n_trials=n_trials,
            duration_s=30.0,
            trajectory_pattern=pattern,
            algorithm="my_algorithm",
            base_seed=base_seed,
            scenario="urban_canyon",
            weather_mode="clear",
            packet_profile="critical",
        )
        a3 = ev.run_monte_carlo(
            n_trials=n_trials,
            duration_s=30.0,
            trajectory_pattern=pattern,
            algorithm="a3",
            base_seed=base_seed + 10_000,
            scenario="urban_canyon",
            weather_mode="clear",
            packet_profile="critical",
        )
        return {
            "label": label,
            "proposed_external_aoi_mean": prop["avg_aoi_external_mean"],
            "a3_external_aoi_mean": a3["avg_aoi_external_mean"],
        }
    finally:
        restore()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=10, help="Trials per arm (keep small for quick runs)")
    p.add_argument("--pattern", type=str, default="linear", help="Trajectory pattern")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Snapshot originals
    eta0 = sim_main.PATH_LOSS_EXPONENT
    ghost0 = sim_main.GHOST_NOISE_M
    out0 = sim_main.OUTAGE_RSS_THRESHOLD_DBM
    ws0 = sim_main.SEMANTIC_W_SURVIVAL
    wl0 = sim_main.SEMANTIC_W_LOAD
    cfg_orig = mcr.CONFIG
    rows: List[Dict[str, Any]] = []

    arms: List[Tuple[str, Callable[[], None], Callable[[], None]]] = [
        (
            "baseline_defaults",
            lambda: None,
            lambda: None,
        ),
        (
            "path_loss_eta=2.0",
            lambda: setattr(sim_main, "PATH_LOSS_EXPONENT", 2.0),
            lambda: setattr(sim_main, "PATH_LOSS_EXPONENT", eta0),
        ),
        (
            "path_loss_eta=2.8",
            lambda: setattr(sim_main, "PATH_LOSS_EXPONENT", 2.8),
            lambda: setattr(sim_main, "PATH_LOSS_EXPONENT", eta0),
        ),
        (
            "ghost_noise_m=0.9",
            lambda: setattr(sim_main, "GHOST_NOISE_M", 0.9),
            lambda: setattr(sim_main, "GHOST_NOISE_M", ghost0),
        ),
        (
            "ghost_noise_m=3.5",
            lambda: setattr(sim_main, "GHOST_NOISE_M", 3.5),
            lambda: setattr(sim_main, "GHOST_NOISE_M", ghost0),
        ),
        (
            "outage_threshold=-90dBm",
            lambda: setattr(sim_main, "OUTAGE_RSS_THRESHOLD_DBM", -90.0),
            lambda: setattr(sim_main, "OUTAGE_RSS_THRESHOLD_DBM", out0),
        ),
        (
            "outage_threshold=-100dBm",
            lambda: setattr(sim_main, "OUTAGE_RSS_THRESHOLD_DBM", -100.0),
            lambda: setattr(sim_main, "OUTAGE_RSS_THRESHOLD_DBM", out0),
        ),
        (
            "w_survival=0.5",
            lambda: setattr(sim_main, "SEMANTIC_W_SURVIVAL", 0.5),
            lambda: setattr(sim_main, "SEMANTIC_W_SURVIVAL", ws0),
        ),
        (
            "w_load=0.3",
            lambda: setattr(sim_main, "SEMANTIC_W_LOAD", 0.3),
            lambda: setattr(sim_main, "SEMANTIC_W_LOAD", wl0),
        ),
        (
            "mc_fading_off",
            lambda: setattr(mcr, "CONFIG", replace(cfg_orig, monte_carlo_enable_small_scale_fading=False)),
            lambda: setattr(mcr, "CONFIG", cfg_orig),
        ),
    ]

    print("label,proposed_ext_aoi,a3_ext_aoi")
    for label, apply_p, restore_p in arms:
        row = _run_cell(label, apply_p, restore_p, args.trials, args.pattern, args.seed)
        rows.append(row)
        print(
            f"{row['label']},{row['proposed_external_aoi_mean']:.6f},{row['a3_external_aoi_mean']:.6f}"
        )

    out = Path("outputs") / "sensitivity_sweep_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
