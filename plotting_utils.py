from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def _safe_boxplot(ax_or_plt, data, labels, **kwargs):
    """Matplotlib compatibility helper for labels/tick_labels parameter names."""
    try:
        return ax_or_plt.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError:
        return ax_or_plt.boxplot(data, labels=labels, **kwargs)


def plot_cdf_comparison(results_list: List[Dict], metric: str, output_path: str, title: str = None):
    """
    Plot CDF comparison of multiple algorithms.

    results_list: List of dicts from Monte Carlo runs
    metric: 'avg_aoi', 'peak_aoi', 'handoff_count', etc.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]

    for idx, result in enumerate(results_list):
        values = [r[metric] for r in result["all_results"]]
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        mean_val = np.mean(values)
        ci_low, ci_high = np.percentile(values, [2.5, 97.5])

        label = (
            f"{result['algorithm']} "
            f"(mu={mean_val:.3f}, "
            f"95% CI=[{ci_low:.3f}, {ci_high:.3f}])"
        )

        plt.plot(sorted_vals, cdf, label=label, linewidth=2.5, color=colors[idx % len(colors)])

    plt.xlabel(metric.replace("_", " ").title(), fontsize=12)
    plt.ylabel("CDF", fontsize=12)
    plt.title(
        title or f'Cumulative Distribution: {metric.replace("_", " ").title()}',
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"CDF plot saved: {output_path}")


def plot_boxplot_comparison(results_list: List[Dict], metric: str, output_path: str, title: str = None):
    """Create boxplot comparison of multiple algorithms."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))

    data = []
    labels = []

    for result in results_list:
        values = [r[metric] for r in result["all_results"]]
        data.append(values)
        labels.append(f"{result['algorithm']}\n(n={len(values)})")

    bp = _safe_boxplot(
        plt,
        data,
        labels,
        patch_artist=True,
        medianprops=dict(color="red", linewidth=2),
        boxprops=dict(alpha=0.7),
        widths=0.6,
    )

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
    plt.title(
        title or f'Distribution Comparison: {metric.replace("_", " ").title()}',
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3, axis="y", linestyle="--")
    plt.xticks(rotation=0)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Boxplot saved: {output_path}")


def plot_time_series_example(result: Dict, output_path: str, n_samples: int = 1000):
    """Plot time series example from one trial."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    trial = result["all_results"][0]
    t = np.arange(len(trial["aoi_values"])) / 60.0
    t = t[:n_samples]

    axes[0].plot(t, trial["aoi_values"][:n_samples], linewidth=1.5, color="#2E86AB")
    axes[0].set_ylabel("AoI (s)")
    axes[0].set_title("Age of Information Over Time")
    axes[0].grid(True, alpha=0.3)

    tower_name = trial["connected_towers"][0]
    rss_values = trial["rss_values"][tower_name][:n_samples]
    axes[1].plot(t, rss_values, linewidth=1.5, color="#A23B72")
    axes[1].axhline(y=-95, color="red", linestyle="--", alpha=0.5, label="Outage threshold")
    axes[1].set_ylabel("RSS (dBm)")
    axes[1].set_title(f"Connected Tower RSS ({tower_name})")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower left")

    tower_ids = list(trial["rss_values"].keys())
    tower_indices = [tower_ids.index(name) for name in trial["connected_towers"][:n_samples]]
    axes[2].plot(t, tower_indices, linewidth=1.5, color="#F18F01", drawstyle="steps-post")
    axes[2].set_ylabel("Tower ID")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_yticks(range(len(tower_ids)))
    axes[2].set_yticklabels(tower_ids)
    axes[2].set_title("Connected Tower Timeline (Handoffs)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Time series plot saved: {output_path}")


def run_statistical_tests(results_list: List[Dict], output_path: str):
    """
    Run pairwise t-tests between all algorithms.

    Args:
        results_list: List of Monte Carlo results dicts
        output_path: Path to save CSV report
    """
    metrics = ["avg_aoi", "peak_aoi", "handoff_count", "outage_time_s"]
    report = []

    # Group by trajectory pattern first to avoid confounding effects.
    grouped_by_pattern: Dict[str, List[Dict]] = {}
    for result in results_list:
        pattern = result.get("trajectory_pattern", "unknown")
        grouped_by_pattern.setdefault(pattern, []).append(result)

    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS (Two-tailed t-test)")
    print("=" * 80)

    for pattern, pattern_results in sorted(grouped_by_pattern.items()):
        data_by_algo_metric: Dict[str, Dict[str, List[float]]] = {}
        for result in pattern_results:
            algo = result["algorithm"]
            data_by_algo_metric.setdefault(algo, {m: [] for m in metrics})
            for metric in metrics:
                data_by_algo_metric[algo][metric].extend([trial[metric] for trial in result["all_results"]])
        algorithms = sorted(data_by_algo_metric.keys())

        print(f"\nPattern: {pattern}")
        print("-" * 80)

        for metric in metrics:
            print(f"\n{metric.upper()}:")
            print("-" * 60)

            for i in range(len(algorithms)):
                for j in range(i + 1, len(algorithms)):
                    algo1, algo2 = algorithms[i], algorithms[j]
                    data1 = data_by_algo_metric[algo1][metric]
                    data2 = data_by_algo_metric[algo2][metric]

                    if len(data1) < 2 or len(data2) < 2:
                        t_stat, p_value = 0.0, 1.0
                    elif np.std(data1) == 0.0 and np.std(data2) == 0.0 and np.mean(data1) == np.mean(data2):
                        t_stat, p_value = 0.0, 1.0
                    else:
                        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                        if not np.isfinite(t_stat) or not np.isfinite(p_value):
                            t_stat, p_value = 0.0, 1.0

                    pooled_std = np.sqrt((np.std(data1) ** 2 + np.std(data2) ** 2) / 2)
                    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0.0

                    significance = "SIGNIFICANT" if p_value < 0.05 else "NOT significant"
                    effect_size = "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small"

                    print(f"  {algo1} vs {algo2}:")
                    print(f"    t-statistic: {t_stat:.4f}")
                    print(f"    p-value:     {p_value:.6f} ({significance})")
                    print(f"    Cohen's d:   {cohens_d:.4f} ({effect_size} effect)")

                    report.append(
                        {
                            "trajectory_pattern": pattern,
                            "metric": metric,
                            "algorithm_1": algo1,
                            "algorithm_2": algo2,
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                            "cohens_d": cohens_d,
                            "effect_size": effect_size,
                        }
                    )

    df = pd.DataFrame(report)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print(f"Full report saved to: {output_path}")
    print("=" * 80 + "\n")

    return df


def plot_all_metrics_boxplots(results_list: List[Dict], output_dir: str):
    """
    Generate boxplots for ALL core metrics in one figure.
    """
    metrics = [
        ("avg_aoi", "Average AoI (s)"),
        ("peak_aoi", "Peak AoI (s)"),
        ("handoff_count", "Handoff Count"),
        ("outage_time_s", "Outage Time (s)"),
    ]

    # Aggregate per algorithm label
    grouped: Dict[str, List[Dict]] = {}
    for result in results_list:
        grouped.setdefault(result["algorithm"], []).append(result)
    algorithms = sorted(grouped.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]

    for idx, (metric, title) in enumerate(metrics):
        data = []
        labels = []
        for algo in algorithms:
            values = []
            for result in grouped[algo]:
                values.extend([trial[metric] for trial in result["all_results"]])
            data.append(values)
            labels.append(f"{algo}\n(n={len(values)})")

        ax = axes[idx]
        bp = _safe_boxplot(
            ax,
            data,
            labels,
            patch_artist=True,
            medianprops=dict(color="red", linewidth=2.5),
            boxprops=dict(alpha=0.8),
            widths=0.6,
            showfliers=True,
            flierprops=dict(marker="o", markerfacecolor="red", markersize=4, alpha=0.5),
        )

        for box_idx, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[box_idx % len(colors)])

        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.tick_params(axis="x", rotation=0)

    plt.suptitle("Performance Metrics Comparison Across Algorithms", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "all_metrics_boxplots.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Boxplots saved: {output_path}")
    return output_path
