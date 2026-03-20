from pathlib import Path

from monte_carlo_runner import MonteCarloEvaluator
from plotting_utils import (
    plot_all_metrics_boxplots,
    plot_boxplot_comparison,
    plot_cdf_comparison,
    run_statistical_tests,
)


def run_comprehensive_comparison():
    """
    Run all algorithms and generate comparison plots.
    """
    algorithms = {
        "my_algorithm": "My Algorithm (DT-enabled)",
        "a3": "A3 (3GPP Standard)",
        "greedy": "Greedy RSS",
        "qlearning": "Q-Learning",
    }

    patterns = ["random_walk", "linear", "circular"]
    all_results = []

    for pattern in patterns:
        print(f"\n{'=' * 70}")
        print(f"TRAJECTORY PATTERN: {pattern.upper()}")
        print("=" * 70)

        pattern_results = []
        for algo_name, algo_display in algorithms.items():
            evaluator = MonteCarloEvaluator(base_scenario="urban_canyon")
            result = evaluator.run_monte_carlo(
                n_trials=50,
                duration_s=30.0,
                trajectory_pattern=pattern,
                algorithm=algo_name,
                base_seed=42,
            )
            result["display_name"] = algo_display
            result["algorithm"] = algo_display  # plotting labels
            pattern_results.append(result)
            evaluator.save_results_to_csv(result)

        output_dir = Path("outputs") / f"comparison_{pattern}"
        output_dir.mkdir(parents=True, exist_ok=True)

        for metric in ["avg_aoi", "peak_aoi", "handoff_count"]:
            plot_cdf_comparison(
                pattern_results,
                metric=metric,
                output_path=str(output_dir / f"cdf_{metric}.png"),
                title=f"{metric.replace('_', ' ').title()} - {pattern.replace('_', ' ').title()}",
            )

        plot_boxplot_comparison(
            pattern_results,
            metric="avg_aoi",
            output_path=str(output_dir / "boxplot_aoi.png"),
            title=f"Average AoI Comparison - {pattern.replace('_', ' ').title()}",
        )

        all_results.extend(pattern_results)

    print("\n" + "=" * 70)
    print("ALL COMPARISONS COMPLETE")
    print("=" * 70)
    print(f"Total algorithms tested: {len(algorithms)}")
    print(f"Trajectory patterns: {len(patterns)}")
    print("Results saved to: outputs/")

    print("\n" + "=" * 80)
    print("GENERATING STATISTICAL ANALYSIS")
    print("=" * 80)

    stats_report = run_statistical_tests(
        all_results,
        output_path="outputs/statistical_tests.csv",
    )
    _ = stats_report

    plot_all_metrics_boxplots(
        all_results,
        output_dir="outputs/comparison_all_metrics",
    )

    print("\nALL ANALYSES COMPLETE!")
    print("Check outputs/ for:")
    print("  - statistical_tests.csv (p-values, effect sizes)")
    print("  - comparison_all_metrics/all_metrics_boxplots.png (4-panel comparison)")

    return all_results


if __name__ == "__main__":
    results = run_comprehensive_comparison()
