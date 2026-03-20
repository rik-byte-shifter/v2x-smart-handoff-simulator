import sys

sys.path.append("src")
import main as sim_main
from monte_carlo_runner import MonteCarloEvaluator


def run_controlled_handoff_test() -> None:
    # Temporary stress settings to validate trigger path.
    sim_main.HANDOFF_MARGIN_DB = 1.5
    sim_main.HANDOFF_HOLD_S = 0.15

    evaluator = MonteCarloEvaluator(base_scenario="urban_canyon")
    result = evaluator.run_single_trial(
        trial_id=0,
        duration_s=30.0,
        trajectory_pattern="linear",
        algorithm="my_algorithm",
        seed=123,
        debug_mode=True,
    )

    print("=== CONTROLLED HANDOFF TEST ===")
    print(f"Handoffs: {result['handoff_count']}")
    print(f"Ping-pong: {result['ping_pong_count']}")
    print(f"Avg AoI: {result['avg_aoi']:.4f}s")
    print(f"Peak AoI: {result['peak_aoi']:.4f}s")
    print(f"Outage: {result['outage_time_s']:.4f}s")
    print(f"Avg RSS: {result['avg_rss_dbm']:.2f} dBm")

    if result["handoff_count"] == 0:
        print("Result: 0 handoffs in controlled mode -> investigate trigger conditions further.")
    else:
        print("Result: handoff logic is active under controlled conditions.")


if __name__ == "__main__":
    run_controlled_handoff_test()
