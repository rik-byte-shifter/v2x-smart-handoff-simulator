"""
Genetic-algorithm search for semantic score weights (optional paper / appendix).

Fitness uses the same Monte Carlo harness as the rest of the project, with
``policy_weight_override`` so module defaults in ``src/main.py`` stay unchanged.

Example:
  python semantic_weight_tuner.py --generations 5 --population 8 --trials-per-eval 4
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Dict, List, Tuple

from config import CONFIG

sys.path.append("src")
from monte_carlo_runner import MonteCarloEvaluator  # noqa: E402

KEYS = ["w_survival", "w_pred_rss", "w_load", "w_aoi"]
BOUNDS: Dict[str, Tuple[float, float]] = {
    "w_survival": (0.15, 2.8),
    "w_pred_rss": (0.05, 0.55),
    "w_load": (0.1, 1.35),
    "w_aoi": (0.15, 1.55),
}


def random_genome(rng: random.Random) -> Dict[str, float]:
    return {k: rng.uniform(*BOUNDS[k]) for k in KEYS}


def crossover(rng: random.Random, a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    return {k: a[k] if rng.random() < 0.5 else b[k] for k in KEYS}


def mutate(rng: random.Random, g: Dict[str, float], rate: float = 0.28) -> Dict[str, float]:
    out = dict(g)
    for k in KEYS:
        if rng.random() < rate:
            lo, hi = BOUNDS[k]
            span = hi - lo
            out[k] = max(lo, min(hi, out[k] + rng.gauss(0.0, span * 0.11)))
    return out


def evaluate(
    evaluator: MonteCarloEvaluator,
    genome: Dict[str, float],
    n_trials: int,
    base_seed: int,
    trajectory_pattern: str,
) -> float:
    stats = evaluator.run_monte_carlo(
        n_trials=n_trials,
        duration_s=CONFIG.monte_carlo_duration_s,
        trajectory_pattern=trajectory_pattern,
        algorithm="my_algorithm",
        base_seed=base_seed,
        scenario="urban_canyon",
        weather_mode="clear",
        packet_profile="critical",
        policy_weight_override=genome,
    )
    return (
        float(stats["avg_aoi_external_mean"])
        + 0.028 * float(stats["handoff_count_mean"])
        + 0.09 * float(stats["ping_pong_count_mean"])
    )


def run_ga(
    population: int,
    generations: int,
    trials_per_eval: int,
    seed: int,
    trajectory_pattern: str,
) -> Tuple[Dict[str, float], float]:
    rng = random.Random(seed)
    evaluator = MonteCarloEvaluator(base_scenario="urban_canyon")
    pop: List[Dict[str, float]] = [random_genome(rng) for _ in range(population)]
    best_g: Dict[str, float] = dict(pop[0])
    best_f = float("inf")

    for gen in range(generations):
        scored: List[Tuple[float, Dict[str, float]]] = []
        for i, g in enumerate(pop):
            f = evaluate(
                evaluator,
                g,
                trials_per_eval,
                base_seed=9000 + gen * population + i,
                trajectory_pattern=trajectory_pattern,
            )
            scored.append((f, g))
            if f < best_f:
                best_f = f
                best_g = dict(g)
            print(f"gen {gen:02d} ind {i:02d} fitness={f:.6f}  {g}")

        scored.sort(key=lambda x: x[0])
        elite_n = max(2, population // 2)
        survivors = [dict(g) for _, g in scored[:elite_n]]
        pop = list(survivors)
        while len(pop) < population:
            p1, p2 = rng.sample(survivors, 2)
            pop.append(mutate(rng, crossover(rng, p1, p2)))

    return best_g, best_f


def main() -> None:
    p = argparse.ArgumentParser(description="GA tuner for semantic handoff weights.")
    p.add_argument("--population", type=int, default=10)
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--trials-per-eval", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trajectory", type=str, default="random_walk")
    args = p.parse_args()

    best_g, best_f = run_ga(
        population=args.population,
        generations=args.generations,
        trials_per_eval=args.trials_per_eval,
        seed=args.seed,
        trajectory_pattern=args.trajectory,
    )
    print("\n=== Best (min fitness) ===")
    print(f"fitness={best_f:.6f}")
    print("policy_weight_override / apply to main.py defaults if adopting:")
    for k in KEYS:
        print(f"  {k}: {best_g[k]:.4f}")


if __name__ == "__main__":
    main()
