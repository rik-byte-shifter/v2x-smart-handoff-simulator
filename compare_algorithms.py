"""
Algorithm comparison: full factorial over scenario × weather × packet profile × trajectory,
or a compact legacy mode (single environment).
"""

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu

from monte_carlo_runner import MonteCarloEvaluator
from plotting_utils import (
    _extract_trial_metric_values,
    _holm_bonferroni,
    plot_all_metrics_boxplots,
    plot_boxplot_comparison,
    plot_cdf_comparison,
    run_statistical_tests,
    run_targeted_proposed_vs_a3_all_cells,
)

# Full factorial grid (see README / paper methods).
SCENARIOS = ["baseline", "urban_canyon", "intersection"]
WEATHER_MODES = ["clear", "rain", "fog", "storm"]
PACKET_PROFILES = ["critical", "standard"]
TRAJECTORY_PATTERNS = ["random_walk", "linear", "circular", "manhattan_grid", "highway"]

# Trial counts: more trials on **primary** cells (urban_canyon × all trajectory patterns)
# for power on pre-specified contrasts; fewer elsewhere to cap factorial runtime.
PRIMARY_TRIALS = 100
SECONDARY_TRIALS = 25
# Backward-compatible name: exploratory / non-primary cells use this count.
DEFAULT_FACTORIAL_N_TRIALS = SECONDARY_TRIALS

PRIMARY_TRIAL_SCENARIOS = frozenset({"urban_canyon"})
PRIMARY_TRIAL_PATTERNS = frozenset({"random_walk", "linear", "circular", "manhattan_grid", "highway"})


def _n_trials_for_factorial_cell(
    scenario: Optional[str], pattern: str, uniform_override: Optional[int]
) -> int:
    if uniform_override is not None:
        return uniform_override
    if scenario in PRIMARY_TRIAL_SCENARIOS and pattern in PRIMARY_TRIAL_PATTERNS:
        return PRIMARY_TRIALS
    return SECONDARY_TRIALS

# Plots and legacy-style stats focus on this subset so figures stay interpretable.
DEFAULT_PLOT_FILTER: Tuple[str, str, str] = ("urban_canyon", "clear", "critical")

# Pre-specified **primary inferential family**: Holm–Bonferroni applies across this set only
# (not across all cells in ``statistical_tests.csv``). Aligns with DEFAULT_PLOT_FILTER env.
# Tuple: (metric_alias, experiment_scenario, trajectory_pattern, algo1_label, algo2_label).
# Metric aliases: ``external_aoi`` → trial field ``avg_aoi_external``. Algorithm labels:
# ``Proposed`` → ``my_algorithm``; ``Robust_A3+`` / ``RobustA3`` → ``robust_a3``.
PRIMARY_ENDPOINT_ENV: Tuple[str, str] = ("clear", "critical")
PRIMARY_ENDPOINTS: List[Tuple[str, str, str, str, str]] = [
    ("external_aoi", "urban_canyon", "random_walk", "Proposed", "Robust_A3+"),
    ("external_aoi", "urban_canyon", "linear", "Proposed", "Robust_A3+"),
    ("external_aoi", "urban_canyon", "circular", "Proposed", "Robust_A3+"),
]

# All other comparisons (other metrics, scenarios, full ``statistical_tests.csv`` grid) are
# **exploratory** unless promoted into PRIMARY_ENDPOINTS; report raw p-values without a
# study-wide FWER claim outside this family.
SECONDARY_ENDPOINTS: List[Tuple[str, str, str, str, str]] = []

COUNT_METRICS_PRIMARY = frozenset({"handoff_count", "ping_pong_count"})


def _resolve_primary_metric(alias: str) -> str:
    key = "".join(c for c in alias.lower() if c.isalnum())
    if key in ("externalaoi", "avgaoiexternal"):
        return "avg_aoi_external"
    return alias


def _resolve_primary_algo_key(label: str) -> str:
    """Map short paper labels to ``algorithm_key`` used in Monte Carlo results."""
    if label in (
        "my_algorithm",
        "a3",
        "robust_a3",
        "mpc",
        "greedy",
        "qlearning",
    ):
        return label
    norm = "".join(c for c in label.lower() if c.isalnum())
    if norm == "proposed":
        return "my_algorithm"
    if norm in ("robusta3", "robusta3plus"):
        return "robust_a3"
    raise ValueError(f"Unknown primary-endpoint algorithm label: {label!r}")


def run_primary_tests(results_list: List[Dict]) -> Dict:
    """
    Run only the pre-specified primary family and apply Holm–Bonferroni **across that
    family** (m = len(PRIMARY_ENDPOINTS)), not across the full factorial pairwise grid.
    """
    w_env, p_env = PRIMARY_ENDPOINT_ENV
    primary_tests: List[Dict] = []
    primary_p_values: List[float] = []

    for metric_alias, scenario, pattern, algo1_lbl, algo2_lbl in PRIMARY_ENDPOINTS:
        metric = _resolve_primary_metric(metric_alias)
        k1 = _resolve_primary_algo_key(algo1_lbl)
        k2 = _resolve_primary_algo_key(algo2_lbl)
        d1 = _extract_trial_metric_values(
            results_list, k1, scenario, pattern, w_env, p_env, metric
        )
        d2 = _extract_trial_metric_values(
            results_list, k2, scenario, pattern, w_env, p_env, metric
        )

        if len(d1) < 2 or len(d2) < 2:
            test_type = "insufficient_n"
            stat = float("nan")
            p_value = 1.0
        elif metric in COUNT_METRICS_PRIMARY:
            test_type = "mannwhitneyu"
            try:
                mw = mannwhitneyu(d1, d2, alternative="two-sided")
                stat, p_value = float(mw.statistic), float(mw.pvalue)
            except ValueError:
                stat, p_value = 0.0, 1.0
            if not (math.isfinite(stat) and math.isfinite(p_value)):
                stat, p_value = 0.0, 1.0
        else:
            test_type = "welch_t"
            t_stat, p_value = stats.ttest_ind(d1, d2, equal_var=False)
            stat = float(t_stat)
            p_value = float(p_value)
            if not (math.isfinite(stat) and math.isfinite(p_value)):
                stat, p_value = 0.0, 1.0

        primary_p_values.append(p_value)
        primary_tests.append(
            {
                "metric": metric,
                "metric_alias": metric_alias,
                "experiment_scenario": scenario,
                "weather_mode": w_env,
                "packet_profile": p_env,
                "trajectory_pattern": pattern,
                "algorithm_1_key": k1,
                "algorithm_2_key": k2,
                "algorithm_1_label": algo1_lbl,
                "algorithm_2_label": algo2_lbl,
                "n_1": len(d1),
                "n_2": len(d2),
                "test_type": test_type,
                "statistic": stat,
                "p_value": p_value,
            }
        )

    corrected, reject = _holm_bonferroni(primary_p_values, alpha=0.05)
    for i, row in enumerate(primary_tests):
        row["p_value_corrected_holm"] = corrected[i] if i < len(corrected) else 1.0
        row["significant_corrected_holm"] = reject[i] if i < len(reject) else False

    return {
        "primary_tests": primary_tests,
        "family_wise_error_rate": 0.05,
        "correction_method": "Holm-Bonferroni",
        "num_comparisons": len(primary_tests),
    }


# Paper-friendly labels for subgroup_analysis_table.csv
SCENARIO_DISPLAY = {
    "baseline": "Baseline",
    "urban_canyon": "Urban Canyon",
    "intersection": "Intersection",
}
WEATHER_DISPLAY = {"clear": "Clear", "rain": "Rain", "fog": "Fog", "storm": "Storm"}
PROFILE_DISPLAY = {"critical": "Critical", "standard": "Standard"}


def _matches_plot_filter(result: Dict, scenario: str, weather: str, profile: str) -> bool:
    return (
        result.get("experiment_scenario") == scenario
        and result.get("weather_mode") == weather
        and result.get("packet_profile") == profile
    )


def write_subgroup_proposed_table(rows: List[Dict], output_path: str) -> Path:
    """Subgroup analysis: proposed policy AoI vs scenario / weather / profile / trajectory."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return path
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def build_subgroup_analysis_table(all_results: List[Dict]) -> pd.DataFrame:
    """
    Paper-style subgroup table: Proposed vs A3 (3GPP) mean AoI, aggregated over
    trajectory patterns within each (scenario, weather, profile) cell.

    Improvement = (A3 − Proposed) / A3 × 100% (lower AoI is better; shown as ``NN% ↓``).
    """
    # (scenario, weather, profile) -> algorithm_key -> list of avg_aoi_mean (one per trajectory)
    by_cell: Dict[Tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for r in all_results:
        key = r.get("algorithm_key")
        if key not in ("my_algorithm", "a3"):
            continue
        m = r.get("avg_aoi_mean")
        if m is None or not isinstance(m, (int, float)):
            continue
        sc = r.get("experiment_scenario")
        if sc is None:
            continue
        w = r.get("weather_mode")
        p = r.get("packet_profile")
        cell = (sc, w, p)
        by_cell[cell][key].append(float(m))

    rows: List[Dict] = []
    for cell in sorted(by_cell.keys(), key=lambda c: (str(c[0]), str(c[1]), str(c[2]))):
        pv = by_cell[cell].get("my_algorithm", [])
        av = by_cell[cell].get("a3", [])
        if not pv or not av:
            continue
        proposed_mean = sum(pv) / len(pv)
        a3_mean = sum(av) / len(av)
        if a3_mean > 1e-12:
            imp_pct = (a3_mean - proposed_mean) / a3_mean * 100.0
        else:
            imp_pct = float("nan")
        sc, w, p = cell
        imp_str = f"{imp_pct:.0f}% ↓" if math.isfinite(imp_pct) else ""

        rows.append(
            {
                "Scenario": SCENARIO_DISPLAY.get(sc, str(sc)),
                "Weather": WEATHER_DISPLAY.get(w, w if w else ""),
                "Profile": PROFILE_DISPLAY.get(p, p if p else ""),
                "Proposed_AoI_s": round(proposed_mean, 6),
                "A3_AoI_s": round(a3_mean, 6),
                "Proposed_AoI": f"{proposed_mean:.3f}s",
                "A3_AoI": f"{a3_mean:.3f}s",
                "Improvement_pct": round(imp_pct, 2) if math.isfinite(imp_pct) else "",
                "Improvement": imp_str,
            }
        )

    return pd.DataFrame(rows)


def run_comprehensive_comparison(
    full_factorial: bool = True,
    n_trials: Optional[int] = None,
    base_seed: int = 42,
    plot_filter: Optional[Tuple[str, str, str]] = DEFAULT_PLOT_FILTER,
) -> List[Dict]:
    """
    Run all algorithms across trajectory patterns and, if ``full_factorial`` is True,
    across scenarios, weather modes, and packet profiles.

    By default, **urban_canyon** cells use ``PRIMARY_TRIALS`` and all other scenarios
    use ``SECONDARY_TRIALS`` (better power on the pre-specified primary family). Pass
    ``n_trials`` to force the same count in every cell (e.g. quick runs).

    Writes ``outputs/subgroup_proposed_aoi.csv`` and ``outputs/subgroup_analysis_table.csv``
    (Proposed vs A3, improvement %) when results are available.

    Statistical tests group by (pattern, scenario, weather, profile) when factorial
    metadata is present. A **pre-specified primary family** (``PRIMARY_ENDPOINTS``)
    gets Holm correction across that family only and is written to
    ``outputs/primary_family_tests.csv``; the full ``statistical_tests.csv`` grid is
    exploratory outside that family. Summary plots use ``plot_filter`` (default:
    urban_canyon / clear / critical) so boxplots are not mixed across 24 environment cells.
    """
    algorithms = {
        "my_algorithm": "Proposed (Survival-Aware)",
        "a3": "A3 (3GPP Standard)",
        "robust_a3": "Robust A3+ (load-aware RSS + TTT)",
        "mpc": "MPC lookahead (RSS + handoff penalty)",
        "greedy": "Greedy RSS",
        "qlearning": "Q-Learning",
    }

    all_results: List[Dict] = []
    subgroup_rows: List[Dict] = []

    if not full_factorial:
        patterns = TRAJECTORY_PATTERNS
        envs: List[Tuple[Optional[str], Optional[str], Optional[str]]] = [
            ("urban_canyon", None, None),
        ]
    else:
        patterns = TRAJECTORY_PATTERNS
        envs = [(s, w, p) for s in SCENARIOS for w in WEATHER_MODES for p in PACKET_PROFILES]

    total_cells = len(patterns) * len(envs)
    cell_idx = 0

    for pattern in patterns:
        for scenario, weather, profile in envs:
            cell_idx += 1
            cell_n_trials = _n_trials_for_factorial_cell(scenario, pattern, n_trials)
            print(f"\n{'=' * 70}")
            print(
                f"CELL {cell_idx}/{total_cells} | pattern={pattern} | "
                f"scenario={scenario} | weather={weather} | profile={profile} | "
                f"n_trials={cell_n_trials}"
            )
            print("=" * 70)

            pattern_results = []
            evaluator = MonteCarloEvaluator(base_scenario=scenario or "urban_canyon")
            for algo_name, algo_display in algorithms.items():
                result = evaluator.run_monte_carlo(
                    n_trials=cell_n_trials,
                    duration_s=30.0,
                    trajectory_pattern=pattern,
                    algorithm=algo_name,
                    base_seed=base_seed,
                    scenario=scenario,
                    weather_mode=weather,
                    packet_profile=profile,
                )
                result["display_name"] = algo_display
                result["algorithm"] = algo_display
                result["algorithm_key"] = algo_name
                result["experiment_scenario"] = result.get("experiment_scenario")
                result["weather_mode"] = result.get("weather_mode")
                result["packet_profile"] = result.get("packet_profile")
                pattern_results.append(result)
                evaluator.save_results_to_csv(result)
                all_results.append(result)

            proposed = next((r for r in pattern_results if r.get("algorithm_key") == "my_algorithm"), None)
            if proposed is not None:
                subgroup_rows.append(
                    {
                        "scenario": scenario,
                        "weather": weather,
                        "profile": profile,
                        "trajectory_pattern": pattern,
                        "proposed_avg_aoi_mean": proposed["avg_aoi_mean"],
                        "proposed_avg_aoi_ci_low": proposed["avg_aoi_ci_low"],
                        "proposed_avg_aoi_ci_high": proposed["avg_aoi_ci_high"],
                        "proposed_avg_aoi_external_mean": proposed.get("avg_aoi_external_mean", float("nan")),
                        "proposed_peak_aoi_mean": proposed["peak_aoi_mean"],
                        "n_trials": cell_n_trials,
                    }
                )

            out_dir = Path("outputs") / "comparison_factorial" / pattern
            if full_factorial:
                tag = f"{scenario}_{weather}_{profile}"
            else:
                tag = "default_env"
            out_dir = out_dir / tag
            out_dir.mkdir(parents=True, exist_ok=True)

            for metric in ["avg_aoi", "peak_aoi", "handoff_count"]:
                plot_cdf_comparison(
                    pattern_results,
                    metric=metric,
                    output_path=str(out_dir / f"cdf_{metric}.png"),
                    title=f"{metric.replace('_', ' ').title()} — {pattern} / {tag}",
                )

            plot_boxplot_comparison(
                pattern_results,
                metric="avg_aoi",
                output_path=str(out_dir / "boxplot_aoi.png"),
                title=f"Average AoI — {pattern} / {tag}",
            )

    if subgroup_rows:
        write_subgroup_proposed_table(subgroup_rows, "outputs/subgroup_proposed_aoi.csv")

    if all_results:
        subgroup_df = build_subgroup_analysis_table(all_results)
        out_tbl = Path("outputs") / "subgroup_analysis_table.csv"
        out_tbl.parent.mkdir(parents=True, exist_ok=True)
        subgroup_df.to_csv(out_tbl, index=False)

    print("\n" + "=" * 70)
    print("ALL COMPARISON CELLS COMPLETE")
    print("=" * 70)
    print(f"Trajectory patterns: {len(patterns)}")
    print(f"Environment cells per pattern: {len(envs)}")
    print(f"Algorithms: {len(algorithms)}")
    print("Subgroup detail: outputs/subgroup_proposed_aoi.csv")
    print("Subgroup vs A3:  outputs/subgroup_analysis_table.csv")

    results_for_plots = all_results
    if plot_filter and full_factorial:
        sc, w, p = plot_filter
        results_for_plots = [r for r in all_results if _matches_plot_filter(r, sc, w, p)]
        if not results_for_plots:
            results_for_plots = all_results

    print("\n" + "=" * 80)
    print("GENERATING STATISTICAL ANALYSIS (all factorial cells)")
    print("=" * 80)

    stats_report = run_statistical_tests(
        all_results,
        output_path="outputs/statistical_tests.csv",
    )
    _ = stats_report

    print("\n" + "=" * 80)
    print("TARGETED TESTS: Proposed vs A3 (pre-specified contrast; trial-level, pooled trajectories)")
    print("=" * 80)
    _ = run_targeted_proposed_vs_a3_all_cells(
        all_results,
        output_path="outputs/targeted_proposed_vs_a3_tests.csv",
    )

    print("\n" + "=" * 80)
    print(
        "PRIMARY FAMILY (pre-specified): Holm across "
        f"{len(PRIMARY_ENDPOINTS)} tests only → outputs/primary_family_tests.csv"
    )
    print("=" * 80)
    primary_summary = run_primary_tests(all_results)
    primary_path = Path("outputs") / "primary_family_tests.csv"
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(primary_summary["primary_tests"]).to_csv(primary_path, index=False)
    print(
        f"  FWER={primary_summary['family_wise_error_rate']}, "
        f"method={primary_summary['correction_method']}, m={primary_summary['num_comparisons']}"
    )
    for row in primary_summary["primary_tests"]:
        sig = "yes" if row.get("significant_corrected_holm") else "no"
        print(
            f"  {row['trajectory_pattern']}: p={row['p_value']:.4g}, "
            f"p_holm={row['p_value_corrected_holm']:.4g}, sig(α=0.05)={sig}"
        )

    print("\n" + "=" * 80)
    print(f"SUMMARY PLOTS (filter: scenario={plot_filter[0]}, weather={plot_filter[1]}, profile={plot_filter[2]})")
    print("=" * 80)

    plot_all_metrics_boxplots(
        results_for_plots,
        output_dir="outputs/comparison_all_metrics",
    )

    print("\nALL ANALYSES COMPLETE!")
    print("Check outputs/ for:")
    print("  - statistical_tests.csv (pairwise tests per pattern × environment cell; exploratory outside primary family)")
    print("  - primary_family_tests.csv (pre-specified primary endpoint family; Holm across this set only)")
    print("  - targeted_proposed_vs_a3_tests.csv (Proposed vs A3 only; aligns with subgroup table)")
    print("  - subgroup_proposed_aoi.csv (per-trajectory proposed AoI)")
    print("  - subgroup_analysis_table.csv (Proposed vs A3, mean over trajectories; Improvement %)")
    print("  - comparison_factorial/… (CDFs and boxplots per cell)")
    print("  - comparison_all_metrics/all_metrics_boxplots.png (filtered summary)")

    return all_results


def run_ablation_study():
    """
    Run ablation study for the proposed policy (predictive + survival + semantic AoI terms).
    """
    variants = {
        "my_algorithm": "Proposed (Survival-Aware) — full",
        "no_survival": "w/o survival score",
        "no_aoi": "w/o AoI term",
        "no_load": "w/o load penalty",
        "reactive_only": "Reactive only (no predictive trigger)",
        "no_relay": "w/o V2V relay fallback",
    }
    patterns = ["random_walk", "linear", "circular", "manhattan_grid", "highway"]
    all_results = []

    for pattern in patterns:
        print(f"\n{'=' * 70}")
        print(f"ABLATION TRAJECTORY PATTERN: {pattern.upper()}")
        print("=" * 70)

        pattern_results = []
        evaluator = MonteCarloEvaluator(base_scenario="urban_canyon")
        for algo_key, display_name in variants.items():
            result = evaluator.run_monte_carlo(
                n_trials=50,
                duration_s=30.0,
                trajectory_pattern=pattern,
                algorithm=algo_key,
                base_seed=42,
                scenario="urban_canyon",
                weather_mode="clear",
                packet_profile="critical",
            )
            result["display_name"] = display_name
            result["algorithm"] = display_name
            result["algorithm_key"] = algo_key
            pattern_results.append(result)
            evaluator.save_results_to_csv(result)

        output_dir = Path("outputs") / f"ablation_{pattern}"
        output_dir.mkdir(parents=True, exist_ok=True)

        for metric in ["avg_aoi", "peak_aoi", "handoff_count", "outage_time_s"]:
            plot_cdf_comparison(
                pattern_results,
                metric=metric,
                output_path=str(output_dir / f"cdf_{metric}.png"),
                title=f"Ablation {metric.replace('_', ' ').title()} - {pattern.replace('_', ' ').title()}",
            )

        plot_boxplot_comparison(
            pattern_results,
            metric="avg_aoi",
            output_path=str(output_dir / "boxplot_aoi.png"),
            title=f"Ablation Average AoI - {pattern.replace('_', ' ').title()}",
        )
        all_results.extend(pattern_results)

    run_statistical_tests(all_results, output_path="outputs/statistical_tests_ablation.csv")
    plot_all_metrics_boxplots(all_results, output_dir="outputs/ablation_all_metrics")
    return all_results


if __name__ == "__main__":
    results = run_comprehensive_comparison()
