"""
Algorithm comparison: full factorial over scenario × weather × packet profile × trajectory,
or a compact legacy mode (single environment).

For long runs, use ``quick=True`` or ``python compare_algorithms.py --quick`` (see README).
"""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu

from config import CONFIG
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

# Fast smoke / iteration (~100×+ faster than full factorial): matches primary inference env
# (urban_canyon, clear, critical) and the three trajectories in ``PRIMARY_ENDPOINTS``.
QUICK_TRAJECTORIES = ["random_walk", "linear", "circular"]
QUICK_ENV_CELLS: List[Tuple[str, str, str]] = [("urban_canyon", "clear", "critical")]
QUICK_DEFAULT_TRIALS = 20
QUICK_DEFAULT_DURATION_S = 15.0

# Full grid size (for runtime hints): patterns × scenarios × weathers × profiles
def _full_factorial_num_cells() -> int:
    return (
        len(TRAJECTORY_PATTERNS)
        * len(SCENARIOS)
        * len(WEATHER_MODES)
        * len(PACKET_PROFILES)
    )


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
    """Subgroup detail CSV: proposed policy **external** AoI (primary) + internal AoI (exploratory) per cell."""
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
    Paper-style subgroup table: Proposed vs A3 (3GPP), aggregated over trajectory
    patterns within each (scenario, weather, profile) cell.

    **Primary rows (paper-facing):** mean **external AoI** (``avg_aoi_external_mean``)—
    independent packet process; see ``compute_external_aoi``.

    **Exploratory columns:** internal simulator AoI (``avg_aoi_mean``), coupled to policy
    mechanics—report in supplement only.

    Improvement = (A3 − Proposed) / A3 × 100% (lower AoI is better; ``NN% ↓``).
    """
    by_cell: Dict[Tuple, Dict[str, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))

    for r in all_results:
        key = r.get("algorithm_key")
        if key not in ("my_algorithm", "a3"):
            continue
        ext = r.get("avg_aoi_external_mean")
        internal = r.get("avg_aoi_mean")
        if ext is None or not isinstance(ext, (int, float)):
            continue
        if internal is None or not isinstance(internal, (int, float)):
            internal = float("nan")
        sc = r.get("experiment_scenario")
        if sc is None:
            continue
        w = r.get("weather_mode")
        p = r.get("packet_profile")
        cell = (sc, w, p)
        by_cell[cell][key].append((float(ext), float(internal)))

    rows: List[Dict] = []
    for cell in sorted(by_cell.keys(), key=lambda c: (str(c[0]), str(c[1]), str(c[2]))):
        pairs_p = by_cell[cell].get("my_algorithm", [])
        pairs_a = by_cell[cell].get("a3", [])
        if not pairs_p or not pairs_a:
            continue
        proposed_ext = sum(t[0] for t in pairs_p) / len(pairs_p)
        a3_ext = sum(t[0] for t in pairs_a) / len(pairs_a)
        int_p = [t[1] for t in pairs_p if math.isfinite(t[1])]
        int_a = [t[1] for t in pairs_a if math.isfinite(t[1])]
        proposed_int = sum(int_p) / len(int_p) if int_p else float("nan")
        a3_int = sum(int_a) / len(int_a) if int_a else float("nan")
        if a3_ext > 1e-12:
            imp_ext = (a3_ext - proposed_ext) / a3_ext * 100.0
        else:
            imp_ext = float("nan")
        if math.isfinite(a3_int) and a3_int > 1e-12:
            imp_int = (a3_int - proposed_int) / a3_int * 100.0
        else:
            imp_int = float("nan")
        sc, w, p = cell
        imp_str = f"{imp_ext:.0f}% ↓" if math.isfinite(imp_ext) else ""

        rows.append(
            {
                "Scenario": SCENARIO_DISPLAY.get(sc, str(sc)),
                "Weather": WEATHER_DISPLAY.get(w, w if w else ""),
                "Profile": PROFILE_DISPLAY.get(p, p if p else ""),
                "Proposed_External_AoI_s": round(proposed_ext, 6),
                "A3_External_AoI_s": round(a3_ext, 6),
                "Proposed_External_AoI": f"{proposed_ext:.3f}s",
                "A3_External_AoI": f"{a3_ext:.3f}s",
                "Improvement_External_pct": round(imp_ext, 2) if math.isfinite(imp_ext) else "",
                "Improvement_External": imp_str,
                "Proposed_Internal_AoI_s_exploratory": round(proposed_int, 6),
                "A3_Internal_AoI_s_exploratory": round(a3_int, 6),
                "Improvement_Internal_pct_exploratory": round(imp_int, 2)
                if math.isfinite(imp_int)
                else "",
            }
        )

    return pd.DataFrame(rows)


def run_comprehensive_comparison(
    full_factorial: bool = True,
    n_trials: Optional[int] = None,
    base_seed: int = 42,
    plot_filter: Optional[Tuple[str, str, str]] = DEFAULT_PLOT_FILTER,
    quick: bool = False,
    duration_s: Optional[float] = None,
    slim_baselines: bool = False,
    save_monte_carlo_csv: bool = True,
) -> List[Dict]:
    """
    Run all algorithms across trajectory patterns and, if ``full_factorial`` is True,
    across scenarios, weather modes, and packet profiles.

    ``quick=True``: only ``QUICK_TRAJECTORIES`` × ``QUICK_ENV_CELLS``, default
    ``QUICK_DEFAULT_TRIALS`` and ``QUICK_DEFAULT_DURATION_S`` (override with
    ``n_trials`` / ``duration_s``). For publication use full factorial.

    ``slim_baselines=True``: omit greedy and Q-learning (~2/7 less work per cell).

    By default, **urban_canyon** cells use ``PRIMARY_TRIALS`` and all other scenarios
    use ``SECONDARY_TRIALS``. Pass ``n_trials`` to force the same count in every cell.

    ``save_monte_carlo_csv=False``: skip writing one summary CSV per algorithm per cell
    (``monte_carlo_runner.save_results_to_csv``); factorial runs still keep in-memory
    results for plots and statistics.
    """
    if duration_s is None:
        duration_s = QUICK_DEFAULT_DURATION_S if quick else 30.0

    algorithms = {
        "my_algorithm": "Proposed (Survival-Aware)",
        "a3": "A3 (3GPP Standard)",
        "robust_a3": "Robust A3+ (load-aware RSS + TTT)",
        "mpc": "MPC lookahead (RSS + handoff penalty)",
        "velocity_aided": "Velocity-aided RSS (lookahead + TTT)",
        "greedy": "Greedy RSS",
        "qlearning": "Q-Learning",
    }
    if slim_baselines:
        algorithms.pop("greedy", None)
        algorithms.pop("qlearning", None)

    all_results: List[Dict] = []
    subgroup_rows: List[Dict] = []

    quick_trials: Optional[int] = None
    if quick:
        patterns = list(QUICK_TRAJECTORIES)
        envs = list(QUICK_ENV_CELLS)
        quick_trials = n_trials if n_trials is not None else QUICK_DEFAULT_TRIALS
    elif not full_factorial:
        patterns = TRAJECTORY_PATTERNS
        envs = [
            ("urban_canyon", None, None),
        ]
    else:
        patterns = TRAJECTORY_PATTERNS
        envs = [(s, w, p) for s in SCENARIOS for w in WEATHER_MODES for p in PACKET_PROFILES]

    total_cells = len(patterns) * len(envs)
    n_alg = len(algorithms)
    if quick:
        print(
            f"\n>>> QUICK MODE: {total_cells} cells × {n_alg} algorithms "
            f"× {quick_trials} trials × {duration_s:.0f}s sim "
            f"(vs full factorial {_full_factorial_num_cells()} cells — much faster).\n"
        )
    elif full_factorial:
        print(
            f"\n>>> FULL FACTORIAL: {_full_factorial_num_cells()} cells × {n_alg} algorithms "
            f"(urban_canyon up to {PRIMARY_TRIALS} trials, others {SECONDARY_TRIALS}). "
            "Long wall-clock expected.\n"
        )
    cell_idx = 0

    import pygame
    from main import V2XSmartHandoffSimulator

    evaluator = MonteCarloEvaluator(base_scenario="urban_canyon")

    for pattern in patterns:
        for scenario, weather, profile in envs:
            cell_idx += 1
            if quick:
                cell_n_trials = quick_trials
            else:
                cell_n_trials = _n_trials_for_factorial_cell(scenario, pattern, n_trials)
            print(f"\n{'=' * 70}")
            print(
                f"CELL {cell_idx}/{total_cells} | pattern={pattern} | "
                f"scenario={scenario} | weather={weather} | profile={profile} | "
                f"n_trials={cell_n_trials}"
            )
            print("=" * 70)

            pattern_results = []
            shared_sim = V2XSmartHandoffSimulator(headless=CONFIG.headless_monte_carlo)
            shared_sim.enable_small_scale_fading_override = CONFIG.monte_carlo_enable_small_scale_fading
            try:
                for algo_name, algo_display in algorithms.items():
                    result = evaluator.run_monte_carlo(
                        n_trials=cell_n_trials,
                        duration_s=duration_s,
                        trajectory_pattern=pattern,
                        algorithm=algo_name,
                        base_seed=base_seed,
                        scenario=scenario,
                        weather_mode=weather,
                        packet_profile=profile,
                        sim=shared_sim,
                        pygame_cleanup=False,
                    )
                    result["display_name"] = algo_display
                    result["algorithm"] = algo_display
                    result["algorithm_key"] = algo_name
                    result["experiment_scenario"] = result.get("experiment_scenario")
                    result["weather_mode"] = result.get("weather_mode")
                    result["packet_profile"] = result.get("packet_profile")
                    pattern_results.append(result)
                    if save_monte_carlo_csv:
                        evaluator.save_results_to_csv(result)
                    all_results.append(result)
            finally:
                pygame.quit()

            proposed = next((r for r in pattern_results if r.get("algorithm_key") == "my_algorithm"), None)
            if proposed is not None:
                subgroup_rows.append(
                    {
                        "scenario": scenario,
                        "weather": weather,
                        "profile": profile,
                        "trajectory_pattern": pattern,
                        "proposed_avg_aoi_external_mean": proposed.get(
                            "avg_aoi_external_mean", float("nan")
                        ),
                        "proposed_avg_aoi_external_ci_low": proposed.get(
                            "avg_aoi_external_ci_low", float("nan")
                        ),
                        "proposed_avg_aoi_external_ci_high": proposed.get(
                            "avg_aoi_external_ci_high", float("nan")
                        ),
                        "proposed_avg_aoi_mean_exploratory": proposed["avg_aoi_mean"],
                        "proposed_avg_aoi_ci_low_exploratory": proposed["avg_aoi_ci_low"],
                        "proposed_avg_aoi_ci_high_exploratory": proposed["avg_aoi_ci_high"],
                        "proposed_peak_aoi_mean": proposed["peak_aoi_mean"],
                        "n_trials": cell_n_trials,
                    }
                )

            out_dir = Path("outputs") / "comparison_factorial" / pattern
            if full_factorial or quick:
                tag = f"{scenario}_{weather}_{profile}"
            else:
                tag = "default_env"
            out_dir = out_dir / tag
            out_dir.mkdir(parents=True, exist_ok=True)

            for metric in [
                "avg_aoi_external",
                "peak_aoi_external",
                "avg_aoi",
                "peak_aoi",
                "handoff_count",
            ]:
                plot_cdf_comparison(
                    pattern_results,
                    metric=metric,
                    output_path=str(out_dir / f"cdf_{metric}.png"),
                    title=f"{metric.replace('_', ' ').title()} — {pattern} / {tag}",
                )

            plot_boxplot_comparison(
                pattern_results,
                metric="avg_aoi_external",
                output_path=str(out_dir / "boxplot_aoi_external.png"),
                title=f"Average External AoI (primary) — {pattern} / {tag}",
            )
            plot_boxplot_comparison(
                pattern_results,
                metric="avg_aoi",
                output_path=str(out_dir / "boxplot_aoi_internal_exploratory.png"),
                title=f"Average Internal AoI (exploratory) — {pattern} / {tag}",
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
    if plot_filter and (full_factorial or quick):
        sc, w, p = plot_filter
        results_for_plots = [r for r in all_results if _matches_plot_filter(r, sc, w, p)]
        if not results_for_plots:
            results_for_plots = all_results

    print("\n" + "=" * 80)
    print("GENERATING STATISTICAL ANALYSIS (all factorial cells)")
    print("=" * 80)

    boot_n = CONFIG.bootstrap_n_resamples_quick if quick else None
    stats_report = run_statistical_tests(
        all_results,
        output_path="outputs/statistical_tests.csv",
        primary_output_path="outputs/statistical_tests_primary_endpoints.csv",
        bootstrap_n_resamples=boot_n,
    )
    _ = stats_report

    print("\n" + "=" * 80)
    print("TARGETED TESTS: Proposed vs A3 (pre-specified contrast; trial-level, pooled trajectories)")
    print("=" * 80)
    _ = run_targeted_proposed_vs_a3_all_cells(
        all_results,
        output_path="outputs/targeted_proposed_vs_a3_tests.csv",
        bootstrap_n_resamples=boot_n,
    )

    print("\n" + "=" * 80)
    print(
        "PRIMARY FAMILY (pre-specified): Holm across "
        f"{len(PRIMARY_ENDPOINTS)} tests only -> outputs/primary_family_tests.csv"
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
            f"p_holm={row['p_value_corrected_holm']:.4g}, sig(a=0.05)={sig}"
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
    print("  - statistical_tests.csv (all metrics; exploratory outside primary family except noted)")
    print("  - statistical_tests_primary_endpoints.csv (external AoI + churn + outage only; paper-facing grid)")
    print("  - primary_family_tests.csv (pre-specified primary endpoint family; Holm across this set only)")
    print("  - targeted_proposed_vs_a3_tests.csv (Proposed vs A3 only; aligns with subgroup table)")
    print("  - subgroup_proposed_aoi.csv (per-trajectory proposed AoI)")
    print("  - subgroup_analysis_table.csv (Proposed vs A3; primary = external AoI; internal = exploratory)")
    print("  - comparison_factorial/... (CDFs and boxplots per cell)")
    print("  - comparison_all_metrics/all_metrics_boxplots.png (filtered summary)")

    return all_results


def run_ablation_study(
    n_trials: int = 50,
    duration_s: float = 30.0,
    patterns: Optional[List[str]] = None,
    base_seed: int = 42,
) -> List[Dict]:
    """
    Run ablation study for the proposed policy (predictive + survival + AoI utility terms).

    Defaults match the original paper-style run (50 trials × 30 s × five trajectories).
    For faster appendix smoke tests, pass e.g. ``n_trials=15``, ``duration_s=15.0``,
    ``patterns=["random_walk", "linear", "circular"]``.
    """
    variants = {
        "my_algorithm": "Proposed (Survival-Aware) - full",
        "no_survival": "w/o survival score",
        "no_aoi": "w/o AoI term",
        "no_load": "w/o load penalty",
        "reactive_only": "Reactive only (no predictive trigger)",
        "no_relay": "w/o V2V relay fallback",
    }
    if patterns is None:
        patterns = ["random_walk", "linear", "circular", "manhattan_grid", "highway"]
    all_results = []

    print(
        f"\n>>> ABLATION: {len(patterns)} pattern(s) × {len(variants)} variants × "
        f"{n_trials} trials × {duration_s:.0f}s sim\n"
    )

    import pygame
    from main import V2XSmartHandoffSimulator

    evaluator = MonteCarloEvaluator(base_scenario="urban_canyon")

    for pattern in patterns:
        print(f"\n{'=' * 70}")
        print(f"ABLATION TRAJECTORY PATTERN: {pattern.upper()}")
        print("=" * 70)

        pattern_results = []
        shared_sim = V2XSmartHandoffSimulator(headless=CONFIG.headless_monte_carlo)
        shared_sim.enable_small_scale_fading_override = CONFIG.monte_carlo_enable_small_scale_fading
        try:
            for algo_key, display_name in variants.items():
                result = evaluator.run_monte_carlo(
                    n_trials=n_trials,
                    duration_s=duration_s,
                    trajectory_pattern=pattern,
                    algorithm=algo_key,
                    base_seed=base_seed,
                    scenario="urban_canyon",
                    weather_mode="clear",
                    packet_profile="critical",
                    sim=shared_sim,
                    pygame_cleanup=False,
                )
                result["display_name"] = display_name
                result["algorithm"] = display_name
                result["algorithm_key"] = algo_key
                pattern_results.append(result)
                evaluator.save_results_to_csv(result)
        finally:
            pygame.quit()

        output_dir = Path("outputs") / f"ablation_{pattern}"
        output_dir.mkdir(parents=True, exist_ok=True)

        for metric in [
            "avg_aoi_external",
            "peak_aoi_external",
            "avg_aoi",
            "peak_aoi",
            "handoff_count",
            "outage_time_s",
        ]:
            plot_cdf_comparison(
                pattern_results,
                metric=metric,
                output_path=str(output_dir / f"cdf_{metric}.png"),
                title=f"Ablation {metric.replace('_', ' ').title()} - {pattern.replace('_', ' ').title()}",
            )

        plot_boxplot_comparison(
            pattern_results,
            metric="avg_aoi_external",
            output_path=str(output_dir / "boxplot_aoi_external.png"),
            title=f"Ablation Average External AoI (primary) - {pattern.replace('_', ' ').title()}",
        )
        plot_boxplot_comparison(
            pattern_results,
            metric="avg_aoi",
            output_path=str(output_dir / "boxplot_aoi_internal_exploratory.png"),
            title=f"Ablation Average Internal AoI (exploratory) - {pattern.replace('_', ' ').title()}",
        )
        all_results.extend(pattern_results)

    run_statistical_tests(
        all_results,
        output_path="outputs/statistical_tests_ablation.csv",
        primary_output_path="outputs/statistical_tests_ablation_primary_endpoints.csv",
    )
    plot_all_metrics_boxplots(all_results, output_dir="outputs/ablation_all_metrics")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Factorial algorithm comparison (use --quick for fast iteration).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Small grid: urban_canyon + clear + critical, trajectories random_walk/linear/circular, "
            f"{QUICK_DEFAULT_TRIALS} trials, {QUICK_DEFAULT_DURATION_S:.0f}s sim (override with --n-trials / --duration-s)."
        ),
    )
    parser.add_argument(
        "--no-full-factorial",
        action="store_true",
        help="Legacy compact mode (all patterns, single env) — not the same as --quick.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        metavar="N",
        help="Uniform trial count for every cell (overrides primary/secondary split unless --quick supplies default).",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Simulated seconds per trial (default 30 full, 15 quick).",
    )
    parser.add_argument(
        "--slim-baselines",
        action="store_true",
        help="Drop greedy + Q-learning to save ~2/7 runtime per cell.",
    )
    parser.add_argument(
        "--no-monte-csv",
        action="store_true",
        help="Skip writing per-algorithm monte_carlo_*.csv files to outputs/ (full factorial writes many).",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run appendix ablation study only (urban_canyon / clear / critical); then exit.",
    )
    parser.add_argument(
        "--ablation-trials",
        type=int,
        default=50,
        metavar="N",
        help="Trials per variant per pattern (default 50).",
    )
    parser.add_argument(
        "--ablation-duration",
        type=float,
        default=30.0,
        help="Simulated seconds per trial for ablation (default 30).",
    )
    parser.add_argument(
        "--ablation-patterns",
        type=str,
        default="",
        help="Comma-separated patterns (default: all five trajectories). Example: random_walk,linear,circular",
    )
    args = parser.parse_args()
    if args.ablation:
        ab_patterns: Optional[List[str]] = None
        if args.ablation_patterns.strip():
            ab_patterns = [p.strip() for p in args.ablation_patterns.split(",") if p.strip()]
        run_ablation_study(
            n_trials=args.ablation_trials,
            duration_s=args.ablation_duration,
            patterns=ab_patterns,
        )
        sys.exit(0)

    results = run_comprehensive_comparison(
        full_factorial=not args.no_full_factorial,
        quick=args.quick,
        n_trials=args.n_trials,
        duration_s=args.duration_s,
        slim_baselines=args.slim_baselines,
        save_monte_carlo_csv=not args.no_monte_csv,
    )
