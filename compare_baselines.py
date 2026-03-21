from datetime import datetime
from pathlib import Path
from typing import Dict, List

from monte_carlo_runner import MonteCarloEvaluator
from plotting_utils import plot_boxplot_comparison, plot_cdf_comparison, plot_time_series_example


def compare_algorithms(
    scenario: str = "urban_canyon",
    n_runs: int = 50,
    duration_s: float = 30.0,
    trajectory_pattern: str = "random_walk",
) -> Dict[str, Dict]:
    algorithms = [
        ("Proposed (Survival-Aware)", "my_algorithm"),
        ("A3 (3GPP)", "a3_3gpp"),
        ("Robust A3+", "a3_robust"),
        ("MPC lookahead", "mpc_lookahead"),
        ("Greedy RSS", "greedy_rss"),
        ("Q-Learning", "q_learning"),
    ]

    evaluator = MonteCarloEvaluator(base_scenario=scenario)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    summary: Dict[str, Dict] = {}
    results_list: List[Dict] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 76)
    print(f"Baseline comparison | scenario={scenario} | runs={n_runs} | trajectory={trajectory_pattern}")
    print("=" * 76)

    for display_name, algorithm_key in algorithms:
        result = evaluator.run_monte_carlo(
            n_trials=n_runs,
            duration_s=duration_s,
            trajectory_pattern=trajectory_pattern,
            algorithm=algorithm_key,
            base_seed=42,
            debug_mode=False,
        )
        result["algorithm"] = display_name
        result["algorithm_key"] = algorithm_key
        results_list.append(result)
        summary[display_name] = result
        csv_path = evaluator.save_results_to_csv(result, output_dir=str(output_dir))
        print(f"Saved summary CSV: {csv_path}")

    plot_cdf_comparison(
        results_list,
        metric="avg_aoi",
        output_path=str(output_dir / f"baseline_cdf_avg_aoi_{stamp}.png"),
        title="AoI CDF Comparison Across Baselines",
    )
    plot_boxplot_comparison(
        results_list,
        metric="handoff_count",
        output_path=str(output_dir / f"baseline_box_handoffs_{stamp}.png"),
        title="Handoff Count Distribution Across Baselines",
    )
    plot_time_series_example(
        results_list[0],
        output_path=str(output_dir / f"baseline_timeseries_example_{stamp}.png"),
        n_samples=1200,
    )

    print("\nAlgorithm         Avg AoI (s)   95% CI                Handoffs   Outage (s)")
    print("-" * 76)
    for name, stats in summary.items():
        ci = (stats["avg_aoi_ci_low"], stats["avg_aoi_ci_high"])
        print(
            f"{name:<16} "
            f"{stats['avg_aoi_mean']:<12.4f} "
            f"[{ci[0]:.4f}, {ci[1]:.4f}]   "
            f"{stats['handoff_count_mean']:<8.2f} "
            f"{stats['outage_time_s_mean']:.3f}"
        )

    return summary


if __name__ == "__main__":
    compare_algorithms(
        scenario="urban_canyon",
        n_runs=50,
        duration_s=30.0,
        trajectory_pattern="random_walk",
    )
