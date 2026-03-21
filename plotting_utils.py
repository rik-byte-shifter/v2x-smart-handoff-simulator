from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import CONFIG

# Primary inferential metrics for paper-facing CSV subset (see docs/ANALYSIS_PROTOCOL.md).
PRIMARY_STATISTICAL_METRICS = frozenset(
    {
        "avg_aoi_external",
        "peak_aoi_external",
        "handoff_count",
        "ping_pong_count",
        "outage_time_s",
    }
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import bootstrap, mannwhitneyu


def _safe_boxplot(ax_or_plt, data, labels, **kwargs):
    """Matplotlib compatibility helper for labels/tick_labels parameter names."""
    try:
        return ax_or_plt.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError:
        return ax_or_plt.boxplot(data, labels=labels, **kwargs)


def _holm_bonferroni(p_values: List[float], alpha: float = 0.05):
    """
    Holm-Bonferroni correction without external dependencies.
    Returns:
      - corrected_p_values in original order
      - reject decisions in original order
    """
    m = len(p_values)
    if m == 0:
        return [], []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected_sorted = [1.0] * m
    for rank, (_, p) in enumerate(indexed):
        corrected_sorted[rank] = min(1.0, (m - rank) * p)
    for i in range(1, m):
        corrected_sorted[i] = max(corrected_sorted[i], corrected_sorted[i - 1])
    corrected = [1.0] * m
    reject = [False] * m
    for rank, (orig_idx, _) in enumerate(indexed):
        corrected[orig_idx] = corrected_sorted[rank]
        reject[orig_idx] = corrected_sorted[rank] < alpha
    return corrected, reject


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
        raw = [r.get(metric) for r in result["all_results"]]
        values = [float(v) for v in raw if v is not None and isinstance(v, (int, float)) and np.isfinite(v)]
        if not values:
            continue
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
        raw = [r.get(metric) for r in result["all_results"]]
        values = [float(v) for v in raw if v is not None and isinstance(v, (int, float)) and np.isfinite(v)]
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

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])

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


def _cliffs_delta(data1: List[float], data2: List[float]) -> float:
    """
    Cliff's delta effect size for two groups (nonparametric; suitable for discrete counts).
    Returns (prop(x1 > x2) - prop(x1 < x2)) over all pairs; in [-1, 1].
    """
    x = np.asarray(data1, dtype=float)
    y = np.asarray(data2, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    greater = 0
    less = 0
    for a in x:
        greater += int(np.sum(a > y))
        less += int(np.sum(a < y))
    denom = x.size * y.size
    return (greater - less) / denom if denom else float("nan")


def _cliffs_delta_magnitude_label(d: float) -> str:
    if not np.isfinite(d):
        return ""
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def _bootstrap_mean_diff_ci(
    data1: List[float],
    data2: List[float],
    n_resamples: Optional[int] = None,
    confidence_level: float = 0.95,
    random_state: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for independent-sample mean difference (mean1 - mean2)."""
    if n_resamples is None:
        n_resamples = CONFIG.bootstrap_n_resamples
    d1 = np.asarray(data1, dtype=float)
    d2 = np.asarray(data2, dtype=float)
    if d1.size < 2 or d2.size < 2:
        return (float("nan"), float("nan"))
    if not np.any(np.isfinite(d1)) or not np.any(np.isfinite(d2)):
        return (float("nan"), float("nan"))

    def _diff_mean(*samples: np.ndarray) -> float:
        return float(np.mean(samples[0]) - np.mean(samples[1]))

    try:
        rng = np.random.default_rng(random_state)
        boot_res = bootstrap(
            (d1, d2),
            statistic=_diff_mean,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method="percentile",
            random_state=rng,
        )
        lo, hi = boot_res.confidence_interval
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return (float("nan"), float("nan"))
        return (float(lo), float(hi))
    except (ValueError, RuntimeError):
        return (float("nan"), float("nan"))


def _bootstrap_hodges_lehmann_ci(
    data1: List[float],
    data2: List[float],
    n_resamples: Optional[int] = None,
    confidence_level: float = 0.95,
    random_state: int = 0,
) -> Tuple[float, float, float]:
    """
    Hodges–Lehmann shift (median of pairwise differences, sample 1 minus sample 2)
    and a percentile bootstrap CI, aligned with rank-based two-sample tests.
    """
    if n_resamples is None:
        n_resamples = CONFIG.bootstrap_n_resamples
    d1 = np.asarray(data1, dtype=float)
    d2 = np.asarray(data2, dtype=float)
    if d1.size < 2 or d2.size < 2:
        return (float("nan"), float("nan"), float("nan"))
    if not np.any(np.isfinite(d1)) or not np.any(np.isfinite(d2)):
        return (float("nan"), float("nan"), float("nan"))

    def _hl_stat(*samples: np.ndarray) -> float:
        a, b = samples[0], samples[1]
        return float(np.median(np.subtract.outer(a, b).ravel()))

    hl_point = _hl_stat(d1, d2)
    try:
        rng = np.random.default_rng(random_state)
        boot_res = bootstrap(
            (d1, d2),
            statistic=_hl_stat,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method="percentile",
            random_state=rng,
        )
        lo, hi = boot_res.confidence_interval
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return (hl_point, float("nan"), float("nan"))
        return (hl_point, float(lo), float(hi))
    except (ValueError, RuntimeError):
        return (hl_point, float("nan"), float("nan"))


TARGETED_COUNT_METRICS = frozenset({"handoff_count", "ping_pong_count"})


def _hodges_lehmann_median_diff(x: List[float], y: List[float]) -> float:
    """Median of all pairwise differences (two-sample Hodges–Lehmann shift estimate)."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    diffs = np.subtract.outer(a, b).ravel()
    return float(np.median(diffs))


def _extract_trial_metric_values(
    results_list: List[Dict],
    algorithm_key: str,
    experiment_scenario: str,
    trajectory_pattern: Optional[str],
    weather_mode: Optional[str],
    packet_profile: Optional[str],
    metric: str,
) -> List[float]:
    """Collect per-trial metric values from Monte Carlo ``all_results`` rows."""
    out: List[float] = []
    for r in results_list:
        if r.get("algorithm_key") != algorithm_key:
            continue
        if r.get("experiment_scenario") != experiment_scenario:
            continue
        if trajectory_pattern is not None and r.get("trajectory_pattern") != trajectory_pattern:
            continue
        if weather_mode is not None and r.get("weather_mode") != weather_mode:
            continue
        if packet_profile is not None and r.get("packet_profile") != packet_profile:
            continue
        for trial in r.get("all_results", []):
            if metric not in trial:
                continue
            v = trial[metric]
            if isinstance(v, (int, float)) and np.isfinite(v):
                out.append(float(v))
    return out


def test_proposed_vs_a3(
    results_list: List[Dict],
    metric: str,
    experiment_scenario: str,
    trajectory_pattern: Optional[str] = None,
    weather_mode: Optional[str] = None,
    packet_profile: Optional[str] = None,
) -> Dict:
    """
    Pre-specified contrast: **Proposed** (``my_algorithm``) vs **A3** (``a3``) only.

    Uses trial-level values from each Monte Carlo run's ``all_results``. When
    ``trajectory_pattern`` is ``None``, trials are **pooled across trajectory patterns**
    for the same (scenario, weather, profile) cell—aligned with subgroup summaries that
    aggregate over patterns.

    Returns p-value (Mann–Whitney U for count metrics, Welch t-test for continuous),
    ``effect_estimate`` / ``effect_ci_*`` aligned with that scale (HL + bootstrap CI for
    counts; mean difference + bootstrap mean-diff CI for continuous), plus auxiliary
    ``mean_diff_proposed_minus_a3`` and ``hl_median_diff`` where applicable.
    """
    proposed_data = _extract_trial_metric_values(
        results_list,
        "my_algorithm",
        experiment_scenario,
        trajectory_pattern,
        weather_mode,
        packet_profile,
        metric,
    )
    a3_data = _extract_trial_metric_values(
        results_list,
        "a3",
        experiment_scenario,
        trajectory_pattern,
        weather_mode,
        packet_profile,
        metric,
    )

    n_p, n_a = len(proposed_data), len(a3_data)
    if n_p < 2 or n_a < 2:
        return {
            "metric": metric,
            "experiment_scenario": experiment_scenario,
            "trajectory_pattern": trajectory_pattern,
            "weather_mode": weather_mode,
            "packet_profile": packet_profile,
            "n_proposed": n_p,
            "n_a3": n_a,
            "test_type": "insufficient_n",
            "statistic": float("nan"),
            "p_value": 1.0,
            "significant": False,
            "mean_diff_proposed_minus_a3": float("nan"),
            "bootstrap_ci_diff_low": float("nan"),
            "bootstrap_ci_diff_high": float("nan"),
            "hl_median_diff": float("nan"),
            "effect_estimate": float("nan"),
            "effect_ci_low": float("nan"),
            "effect_ci_high": float("nan"),
            "effect_label": "",
            "cohens_d": float("nan"),
        }

    mean_diff = float(np.mean(proposed_data) - np.mean(a3_data))
    boot_lo, boot_hi = _bootstrap_mean_diff_ci(proposed_data, a3_data)
    hl_md = _hodges_lehmann_median_diff(proposed_data, a3_data)

    pooled_std = np.sqrt((np.std(proposed_data) ** 2 + np.std(a3_data) ** 2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

    if metric in TARGETED_COUNT_METRICS:
        test_type = "mannwhitneyu"
        try:
            mw = mannwhitneyu(proposed_data, a3_data, alternative="two-sided")
            test_stat, p_value = float(mw.statistic), float(mw.pvalue)
        except ValueError:
            test_stat, p_value = 0.0, 1.0
        if not np.isfinite(test_stat) or not np.isfinite(p_value):
            test_stat, p_value = 0.0, 1.0
        eff_est, eff_lo, eff_hi = _bootstrap_hodges_lehmann_ci(proposed_data, a3_data)
        effect_label = "Hodges-Lehmann median diff (bootstrap 95% CI)"
    else:
        test_type = "welch_t"
        t_stat, p_value = stats.ttest_ind(proposed_data, a3_data, equal_var=False)
        test_stat = float(t_stat)
        p_value = float(p_value)
        if not np.isfinite(test_stat) or not np.isfinite(p_value):
            test_stat, p_value = 0.0, 1.0
        eff_est, eff_lo, eff_hi = mean_diff, boot_lo, boot_hi
        effect_label = "Mean difference (bootstrap 95% CI)"

    return {
        "metric": metric,
        "experiment_scenario": experiment_scenario,
        "trajectory_pattern": trajectory_pattern,
        "weather_mode": weather_mode,
        "packet_profile": packet_profile,
        "n_proposed": n_p,
        "n_a3": n_a,
        "test_type": test_type,
        "statistic": test_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "mean_diff_proposed_minus_a3": mean_diff,
        "bootstrap_ci_diff_low": boot_lo,
        "bootstrap_ci_diff_high": boot_hi,
        "hl_median_diff": hl_md,
        "effect_estimate": eff_est,
        "effect_ci_low": eff_lo,
        "effect_ci_high": eff_hi,
        "effect_label": effect_label,
        "cohens_d": cohens_d,
    }


def run_targeted_proposed_vs_a3_all_cells(
    results_list: List[Dict],
    metrics: Optional[List[str]] = None,
    output_path: str = "outputs/targeted_proposed_vs_a3_tests.csv",
) -> pd.DataFrame:
    """
    For every (scenario, weather, profile) cell present in ``results_list``, run
    :func:`test_proposed_vs_a3` for each metric.

    Trials are **pooled across trajectory patterns** (``trajectory_pattern=None``), matching
    how ``subgroup_analysis_table.csv`` summarizes over ``random_walk`` / ``linear`` / ``circular``.
    """
    if metrics is None:
        metrics = [
            "avg_aoi_external",
            "avg_aoi",
            "handoff_count",
            "ping_pong_count",
            "outage_time_s",
        ]

    cells: set = set()
    for r in results_list:
        if r.get("algorithm_key") not in ("my_algorithm", "a3"):
            continue
        sc = r.get("experiment_scenario")
        if sc is None:
            continue
        cells.add((sc, r.get("weather_mode"), r.get("packet_profile")))

    rows: List[Dict] = []
    for sc, w, p in sorted(cells, key=lambda c: (str(c[0]), str(c[1]), str(c[2]))):
        for metric in metrics:
            row = test_proposed_vs_a3(
                results_list,
                metric,
                experiment_scenario=sc,
                trajectory_pattern=None,
                weather_mode=w,
                packet_profile=p,
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Targeted Proposed vs A3 tests saved to: {output_path}")
    return df


def run_statistical_tests(
    results_list: List[Dict],
    output_path: str,
    primary_output_path: Optional[str] = None,
):
    """
    Pairwise comparisons: Mann–Whitney U for count metrics (skewed/discrete),
    Welch's t-test for continuous metrics. Reported effect and bootstrap CI match the
    scale of the test: Hodges–Lehmann median difference (bootstrap CI) for counts,
    mean difference (bootstrap CI) for continuous metrics. Mean-based bootstrap CIs
    are still recorded as ``mean_diff`` / ``bootstrap_ci_diff_*`` for all metrics.

    For discrete count metrics, **Cohen's d is omitted** (NaN): mean/std-based d can
    explode when variances are tiny; use **Cliff's delta** instead (``cliffs_delta``).

    If ``primary_output_path`` is set, a second CSV is written containing only rows
    whose metric is in ``PRIMARY_STATISTICAL_METRICS`` (external AoI + churn + outage).
    """
    metrics = [
        "avg_aoi_external",
        "peak_aoi_external",
        "avg_aoi",
        "peak_aoi",
        "handoff_count",
        "ping_pong_count",
        "outage_time_s",
    ]
    count_metrics = frozenset({"handoff_count", "ping_pong_count"})
    report = []

    # Group by trajectory + optional factorial cell (scenario / weather / profile).
    grouped_by_cell: Dict[tuple, List[Dict]] = {}
    for result in results_list:
        pattern = result.get("trajectory_pattern", "unknown")
        exp = result.get("experiment_scenario", result.get("base_scenario", ""))
        w = result.get("weather_mode", "")
        p = result.get("packet_profile", "")
        key = (pattern, exp or "", w or "", p or "")
        grouped_by_cell.setdefault(key, []).append(result)

    print("\n" + "=" * 80)
    print(
        "STATISTICAL TESTS: Welch t (continuous) | Mann–Whitney U (counts); "
        "effect CI = bootstrap on mean diff (continuous) or Hodges–Lehmann (counts)"
    )
    print("=" * 80)

    for cell_key, pattern_results in sorted(grouped_by_cell.items()):
        pattern, exp, w, p = cell_key
        data_by_algo_metric: Dict[str, Dict[str, List[float]]] = {}
        for result in pattern_results:
            algo = result["algorithm"]
            data_by_algo_metric.setdefault(algo, {m: [] for m in metrics})
            for metric in metrics:
                for trial in result["all_results"]:
                    if metric not in trial:
                        continue
                    data_by_algo_metric[algo][metric].append(trial[metric])
        algorithms = sorted(data_by_algo_metric.keys())

        print(f"\nPattern: {pattern}")
        if exp or w or p:
            print(f"  Cell: scenario={exp or '-'} | weather={w or '-'} | profile={p or '-'}")
        print("-" * 80)

        for metric in metrics:
            print(f"\n{metric.upper()}:")
            print("-" * 60)

            for i in range(len(algorithms)):
                for j in range(i + 1, len(algorithms)):
                    algo1, algo2 = algorithms[i], algorithms[j]
                    data1 = data_by_algo_metric[algo1][metric]
                    data2 = data_by_algo_metric[algo2][metric]

                    if not data1 or not data2:
                        continue

                    mean_diff = float(np.mean(data1) - np.mean(data2))
                    boot_lo, boot_hi = _bootstrap_mean_diff_ci(data1, data2)

                    if len(data1) < 2 or len(data2) < 2:
                        test_stat, p_value = 0.0, 1.0
                        test_type = "insufficient_n"
                        eff_est = eff_lo = eff_hi = float("nan")
                        effect_label = ""
                    elif (
                        np.std(data1) == 0.0
                        and np.std(data2) == 0.0
                        and np.mean(data1) == np.mean(data2)
                    ):
                        test_stat, p_value = 0.0, 1.0
                        test_type = "degenerate"
                        eff_est = eff_lo = eff_hi = float("nan")
                        effect_label = ""
                    elif metric in count_metrics:
                        test_type = "mannwhitneyu"
                        try:
                            mw = mannwhitneyu(
                                data1,
                                data2,
                                alternative="two-sided",
                            )
                            test_stat, p_value = float(mw.statistic), float(mw.pvalue)
                        except ValueError:
                            test_stat, p_value = 0.0, 1.0
                        if not np.isfinite(test_stat) or not np.isfinite(p_value):
                            test_stat, p_value = 0.0, 1.0
                        eff_est, eff_lo, eff_hi = _bootstrap_hodges_lehmann_ci(
                            data1, data2
                        )
                        effect_label = "Hodges-Lehmann median diff (bootstrap 95% CI)"
                    else:
                        test_type = "welch_t"
                        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                        test_stat = float(t_stat)
                        p_value = float(p_value)
                        if not np.isfinite(test_stat) or not np.isfinite(p_value):
                            test_stat, p_value = 0.0, 1.0
                        eff_est, eff_lo, eff_hi = mean_diff, boot_lo, boot_hi
                        effect_label = "Mean difference (bootstrap 95% CI)"

                    pooled_std = np.sqrt((np.std(data1) ** 2 + np.std(data2) ** 2) / 2)
                    if metric in count_metrics:
                        cohens_d = float("nan")
                        cliffs = _cliffs_delta(data1, data2)
                        effect_size = _cliffs_delta_magnitude_label(cliffs)
                    else:
                        cliffs = float("nan")
                        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
                        effect_size = (
                            "large"
                            if abs(cohens_d) > 0.8
                            else "medium"
                            if abs(cohens_d) > 0.5
                            else "small"
                        )

                    significance = "SIGNIFICANT" if p_value < 0.05 else "NOT significant"

                    print(f"  {algo1} vs {algo2}:")
                    print(f"    test:        {test_type}")
                    print(f"    statistic:   {test_stat:.4f}")
                    print(f"    p-value:     {p_value:.6f} ({significance})")
                    if effect_label and np.isfinite(eff_est):
                        print(
                            f"    effect:      {eff_est:.6f}  "
                            f"95% CI: [{eff_lo:.6f}, {eff_hi:.6f}]  ({effect_label})"
                        )
                    print(
                        f"    mean diff:   {mean_diff:.6f}  "
                        f"95% bootstrap CI (mean): [{boot_lo:.6f}, {boot_hi:.6f}]"
                    )
                    if metric in count_metrics:
                        print(
                            f"    Cliff's d:   {cliffs:.4f} ({effect_size}) "
                            f"(Cohen's d omitted for discrete counts)"
                        )
                    else:
                        print(f"    Cohen's d:   {cohens_d:.4f} ({effect_size} effect)")

                    report.append(
                        {
                            "trajectory_pattern": pattern,
                            "experiment_scenario": exp if exp else None,
                            "weather_mode": w if w else None,
                            "packet_profile": p if p else None,
                            "metric": metric,
                            "algorithm_1": algo1,
                            "algorithm_2": algo2,
                            "test_type": test_type,
                            "t_statistic": test_stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                            "mean_diff": mean_diff,
                            "bootstrap_ci_diff_low": boot_lo,
                            "bootstrap_ci_diff_high": boot_hi,
                            "effect_estimate": eff_est,
                            "effect_ci_low": eff_lo,
                            "effect_ci_high": eff_hi,
                            "effect_label": effect_label,
                            "cohens_d": cohens_d,
                            "cliffs_delta": cliffs,
                            "effect_size": effect_size,
                            "p_value_corrected_holm": 1.0,
                            "significant_corrected_holm": False,
                        }
                    )

    grouped_indices: Dict[tuple, List[int]] = {}
    for idx, row in enumerate(report):
        key = (
            row["trajectory_pattern"],
            row.get("experiment_scenario") or "",
            row.get("weather_mode") or "",
            row.get("packet_profile") or "",
            row["metric"],
        )
        grouped_indices.setdefault(key, []).append(idx)
    for _, idxs in grouped_indices.items():
        pvals = [float(report[i]["p_value"]) for i in idxs]
        corrected, reject = _holm_bonferroni(pvals, alpha=0.05)
        for local_i, global_i in enumerate(idxs):
            report[global_i]["p_value_corrected_holm"] = corrected[local_i]
            report[global_i]["significant_corrected_holm"] = reject[local_i]

    df = pd.DataFrame(report)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if primary_output_path:
        primary_df = df[df["metric"].isin(PRIMARY_STATISTICAL_METRICS)]
        ppath = Path(primary_output_path)
        ppath.parent.mkdir(parents=True, exist_ok=True)
        primary_df.to_csv(ppath, index=False)
        print(f"Primary-endpoint subset saved to: {primary_output_path}")

    print("\n" + "=" * 80)
    print(f"Full report saved to: {output_path}")
    print("=" * 80 + "\n")

    return df


def plot_all_metrics_boxplots(results_list: List[Dict], output_dir: str):
    """
    Generate boxplots for core metrics in one figure (external AoI first, then exploratory internal AoI).
    """
    metrics = [
        ("avg_aoi_external", "Avg external AoI (s) — primary"),
        ("peak_aoi_external", "Peak external AoI (s) — primary"),
        ("avg_aoi", "Avg internal AoI (s) — exploratory"),
        ("peak_aoi", "Peak internal AoI (s) — exploratory"),
        ("handoff_count", "Handoff count"),
        ("outage_time_s", "Outage time (s)"),
    ]

    # Aggregate per algorithm label
    grouped: Dict[str, List[Dict]] = {}
    for result in results_list:
        grouped.setdefault(result["algorithm"], []).append(result)
    algorithms = sorted(grouped.keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]

    for idx, (metric, title) in enumerate(metrics):
        data = []
        labels = []
        for algo in algorithms:
            values = []
            for result in grouped[algo]:
                for trial in result["all_results"]:
                    v = trial.get(metric)
                    if v is not None and isinstance(v, (int, float)) and np.isfinite(float(v)):
                        values.append(float(v))
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
