import csv
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import CONFIG

sys.path.append("src")

# Mirrors src/main.py cycle_weather() entropy map (must stay consistent).
WEATHER_ENTROPY_MAP: Dict[str, float] = {
    "clear": 1.0,
    "rain": 1.3,
    "fog": 1.2,
    "storm": 1.8,
}

# CSV "Algorithm" row and stable filenames when only internal keys are present.
MONTE_CARLO_ALGORITHM_CSV_LABELS: Dict[str, str] = {
    "my_algorithm": "Proposed (Survival-Aware)",
    "a3_3gpp": "A3 (3GPP Standard)",
    "a3_robust": "Robust A3+ (load-aware RSS + TTT)",
    "greedy_rss": "Greedy RSS",
    "q_learning": "Q-Learning",
    "mpc_lookahead": "MPC lookahead (RSS + handoff penalty)",
}


def _monte_carlo_csv_filename_slug(stats_dict: Dict) -> str:
    k = stats_dict.get("algorithm_key")
    if k:
        return str(k)
    a = stats_dict.get("algorithm", "unknown")
    if a in MONTE_CARLO_ALGORITHM_CSV_LABELS:
        return str(a)
    return str(a)


def _monte_carlo_csv_algorithm_label(stats_dict: Dict) -> str:
    k = stats_dict.get("algorithm_key")
    if k and k in MONTE_CARLO_ALGORITHM_CSV_LABELS:
        return MONTE_CARLO_ALGORITHM_CSV_LABELS[k]
    a = stats_dict.get("algorithm", "")
    if a in MONTE_CARLO_ALGORITHM_CSV_LABELS:
        return MONTE_CARLO_ALGORITHM_CSV_LABELS[a]
    return a if a else str(k or "unknown")


def apply_sim_environment(
    sim: "V2XSmartHandoffSimulator",
    weather_mode: Optional[str] = None,
    packet_profile: Optional[str] = None,
) -> None:
    """Set weather entropy and packet urgency profile for a trial (Monte Carlo factorial design)."""
    if weather_mode is not None:
        if weather_mode not in WEATHER_ENTROPY_MAP:
            raise ValueError(f"Unknown weather_mode {weather_mode!r}; expected one of {sorted(WEATHER_ENTROPY_MAP)}")
        sim.weather_mode = weather_mode
        sim.weather_entropy = WEATHER_ENTROPY_MAP[weather_mode]
    if packet_profile is not None:
        if packet_profile not in ("critical", "standard"):
            raise ValueError("packet_profile must be 'critical' or 'standard'")
        sim.packet_profile = packet_profile


import main as sim_main
from baselines import MPCLookaheadBaseline, RobustA3Baseline
from main import (
    FPS,
    HEIGHT,
    METER_PER_PIXEL,
    Measurement,
    OUTAGE_RSS_THRESHOLD_DBM,
    Tower,
    WIDTH,
    V2XSmartHandoffSimulator,
)


def _telemetry_rows_for_external_aoi(trial_results: Dict) -> List[Dict[str, float]]:
    """Build (t, connected_rss) rows; prefer serving RSS trace (relay-safe)."""
    trace = trial_results.get("connected_rss_trace")
    if trace:
        dt = 1.0 / FPS
        return [{"t": i * dt, "connected_rss": float(r)} for i, r in enumerate(trace)]
    rss_values = trial_results.get("rss_values")
    connected = trial_results.get("connected_towers")
    if not rss_values or not connected:
        return []
    dt = 1.0 / FPS
    rows: List[Dict[str, float]] = []
    for i, tname in enumerate(connected):
        per_tower = rss_values.get(tname)
        if not per_tower or i >= len(per_tower):
            break
        rows.append({"t": i * dt, "connected_rss": float(per_tower[i])})
    return rows


def _external_aoi_instantaneous_series(trial_results: Dict, packet_rate_hz: float) -> List[float]:
    telemetry = _telemetry_rows_for_external_aoi(trial_results)
    if not telemetry:
        return []

    last_success_time = 0.0
    next_packet_time = 0.0
    series: List[float] = []
    interval = 1.0 / packet_rate_hz

    for row in telemetry:
        t = float(row["t"])
        connected_rss = float(row["connected_rss"])

        while next_packet_time <= t + 1e-15:
            # Reception success depends only on RSS vs outage threshold (not policy weights).
            success = connected_rss > OUTAGE_RSS_THRESHOLD_DBM
            if success:
                last_success_time = t
            next_packet_time += interval

        series.append(t - last_success_time)

    return series


def compute_external_aoi(trial_results: Dict, packet_rate_hz: float = 10.0) -> float:
    """
    Time-averaged AoI from an external packet process independent of policy weights.

    Packets are generated at fixed ``packet_rate_hz``. Each scheduled reception is
    successful iff serving RSS exceeds ``OUTAGE_RSS_THRESHOLD_DBM`` (same geometry as
    the simulator; no semantic score, load model, or internal AoI integrator).
    """
    series = _external_aoi_instantaneous_series(trial_results, packet_rate_hz)
    if not series:
        return float("nan")
    return float(np.mean(series))


def compute_external_aoi_peak(trial_results: Dict, packet_rate_hz: float = 10.0) -> float:
    """Peak instantaneous external AoI over the trial."""
    series = _external_aoi_instantaneous_series(trial_results, packet_rate_hz)
    if not series:
        return float("nan")
    return float(np.max(series))


class QLearningHandoff:
    """Simple tabular Q-learning baseline policy."""

    def __init__(
        self,
        towers: List[Tower],
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
    ) -> None:
        self.towers = towers
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: Dict[Tuple[int, int, int], Dict[str, float]] = {}

    def _ensure_state(self, state: Tuple[int, int, int]) -> None:
        if state not in self.q_table:
            self.q_table[state] = {t.name: 0.0 for t in self.towers}

    def discretize(self, rss_dbm: float, speed_mps: float, distance_m: float) -> Tuple[int, int, int]:
        rss_bin = int((rss_dbm + 140.0) // 10.0)
        speed_bin = int(speed_mps // 5.0)
        dist_bin = int(distance_m // 50.0)
        return (rss_bin, speed_bin, dist_bin)

    def choose_action(self, state: Tuple[int, int, int]) -> Tower:
        self._ensure_state(state)
        if random.random() < self.epsilon:
            return random.choice(self.towers)
        best_name = max(self.q_table[state], key=self.q_table[state].get)
        return next(t for t in self.towers if t.name == best_name)

    def update(
        self,
        state: Tuple[int, int, int],
        action: Tower,
        reward: float,
        next_state: Tuple[int, int, int],
    ) -> None:
        self._ensure_state(state)
        self._ensure_state(next_state)
        old_val = self.q_table[state][action.name]
        next_max = max(self.q_table[next_state].values())
        self.q_table[state][action.name] = old_val + self.alpha * (
            reward + self.gamma * next_max - old_val
        )


class MonteCarloEvaluator:
    """Run multiple simulation trials with randomized scenarios."""

    def __init__(self, base_scenario: str = "baseline"):
        self.base_scenario = base_scenario
        self.results_history: List[Dict] = []

    def randomize_scenario_parameters(self, sim: V2XSmartHandoffSimulator, seed: int = None) -> None:
        """Add randomness to make each run unique but reproducible."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Randomize UE starting position (within safe zone)
        sim.ue_pos = sim.snap_to_safe_point(
            random.randint(200, WIDTH - 600),
            random.randint(150, HEIGHT - 250),
        )
        sim.prev_ue_pos = sim.ue_pos

        # Randomize blocker phases and slight velocity variations
        for blocker in sim.dynamic_blockers:
            blocker.phase = random.uniform(0, 2 * np.pi)
            blocker.vx *= random.uniform(0.85, 1.15)
            blocker.vy *= random.uniform(0.85, 1.15)
            blocker.x = random.uniform(100, WIDTH - 500)
            blocker.y = random.uniform(100, HEIGHT - 100)

        # Randomize relay positions
        for relay in sim.relays:
            relay.x = random.uniform(200, WIDTH - 600)
            relay.y = random.uniform(150, HEIGHT - 200)
            relay.vx = random.uniform(-15, 15)
            relay.vy = random.uniform(-15, 15)

        # Scheduled load + invisible cross-traffic sessions (other UEs on each tower).
        for tower in sim.towers:
            sim.scheduled_tower_loads[tower.name] = random.uniform(0.12, 0.52)
            sim.background_session_count[tower.name] = random.randint(1, 10)
        sim._sync_cross_load_from_session_counts()
        for tower in sim.towers:
            s = sim.scheduled_tower_loads[tower.name]
            c = sim.cross_traffic_load[tower.name]
            sim.tower_loads[tower.name] = max(0.05, min(0.99, s + c))

    def generate_synthetic_trajectory(
        self, duration_s: float = 30.0, pattern: str = "random_walk"
    ) -> List[Tuple[int, int]]:
        """
        Generate synthetic UE trajectory without mouse input.
        Patterns: 'random_walk', 'linear', 'circular', 'figure8', 'manhattan_grid', 'highway'
        """
        import math

        n_frames = int(duration_s * FPS)
        positions = []
        x, y = WIDTH // 2, HEIGHT // 2

        if pattern == "random_walk":
            vx, vy = random.uniform(-50, 50), random.uniform(-50, 50)
            for _ in range(n_frames):
                vx += random.gauss(0, 5)
                vy += random.gauss(0, 5)

                speed = math.hypot(vx, vy)
                if speed > 35:
                    vx = vx / speed * 35
                    vy = vy / speed * 35

                x += vx / 0.1
                y += vy / 0.1

                if x < 50 or x > WIDTH - 500:
                    vx *= -0.8
                    x = max(50, min(WIDTH - 500, x))
                if y < 50 or y > HEIGHT - 50:
                    vy *= -0.8
                    y = max(50, min(HEIGHT - 50, y))

                positions.append((int(x), int(y)))

        elif pattern == "linear":
            start_x, start_y = 150, HEIGHT // 2
            end_x, end_y = WIDTH - 450, HEIGHT // 2
            for i in range(n_frames):
                t = i / n_frames
                x = int(start_x + (end_x - start_x) * t)
                y = int(start_y + (end_y - start_y) * t + random.gauss(0, 15))
                positions.append((x, y))

        elif pattern == "circular":
            center_x, center_y = WIDTH // 2 - 100, HEIGHT // 2
            radius = 200
            for i in range(n_frames):
                angle = 2 * np.pi * i / n_frames * 2
                x = int(center_x + radius * math.cos(angle))
                y = int(center_y + radius * math.sin(angle))
                positions.append((x, y))

        elif pattern == "figure8":
            center_x, center_y = WIDTH // 2 - 100, HEIGHT // 2
            width, height = 300, 150
            for i in range(n_frames):
                t = 4 * np.pi * i / n_frames
                x = int(center_x + width * math.sin(t))
                y = int(center_y + height * math.sin(2 * t) / 2)
                positions.append((x, y))

        elif pattern == "manhattan_grid":
            margin_l, margin_r = 80, WIDTH - 520
            margin_t, margin_b = 75, HEIGHT - 75
            x, y = float(random.randint(margin_l + 40, margin_l + 120)), float(
                random.uniform(margin_t + 40, margin_b - 80)
            )
            leg_len = max(18, (margin_r - margin_l) // 14)
            direction = 0  # 0=E,1=N,2=W,3=S
            step = 3.8
            for i in range(n_frames):
                if i > 0 and i % leg_len == 0:
                    direction = (direction + random.choice([1, 3])) % 4
                    if random.random() < 0.22:
                        direction = random.randint(0, 3)
                if direction == 0:
                    x += step
                elif direction == 1:
                    y -= step
                elif direction == 2:
                    x -= step
                else:
                    y += step
                if x > margin_r or x < margin_l:
                    x = max(margin_l, min(margin_r, x))
                    direction = 2 if direction == 0 else 0 if direction == 2 else direction
                if y > margin_b or y < margin_t:
                    y = max(margin_t, min(margin_b, y))
                    direction = 3 if direction == 1 else 1 if direction == 3 else direction
                x += random.gauss(0, 0.85)
                y += random.gauss(0, 0.85)
                positions.append((int(max(margin_l, min(margin_r, x))), int(max(margin_t, min(margin_b, y)))))

        elif pattern == "highway":
            vx_mps = random.uniform(20.0, 34.0)
            x = float(random.randint(90, 180))
            y = float(HEIGHT // 2 + random.uniform(-42, 42))
            dt = 1.0 / FPS
            for _ in range(n_frames):
                x += (vx_mps / METER_PER_PIXEL) * dt
                y += random.gauss(0, 1.0)
                y = max(HEIGHT * 0.22, min(HEIGHT * 0.76, y))
                x = max(70.0, min(float(WIDTH - 520), x))
                positions.append((int(x), int(y)))

        else:
            raise ValueError(f"Unsupported trajectory pattern: {pattern}")

        return positions

    def _normalize_algorithm(self, algorithm: str) -> str:
        aliases = {
            "a3": "a3_3gpp",
            "greedy": "greedy_rss",
            "qlearning": "q_learning",
            "handoff_debug": "my_algorithm",
            "full": "my_algorithm",
            "no_survival": "ab_no_survival",
            "no_aoi": "ab_no_aoi",
            "no_load": "ab_no_load",
            "no_prediction": "ab_reactive_only",
            "reactive_only": "ab_reactive_only",
            "no_relay": "ab_no_relay",
            "robust_a3": "a3_robust",
            "a3+": "a3_robust",
            "a3_robust": "a3_robust",
            "mpc": "mpc_lookahead",
            "mpc_lookahead": "mpc_lookahead",
        }
        key = algorithm.lower()
        return aliases.get(key, key)

    def _ablation_profile(self, algorithm_key: str) -> Dict[str, float | bool] | None:
        profiles: Dict[str, Dict[str, float | bool]] = {
            "ab_no_survival": {"w_survival": 0.0},
            "ab_no_aoi": {"w_aoi": 0.0},
            "ab_no_load": {"w_load": 0.0},
            "ab_reactive_only": {"enable_predictive_trigger": False},
            "ab_no_relay": {"enable_relay_fallback": False},
        }
        return profiles.get(algorithm_key)

    def _default_policy_profile_from_module(self) -> Dict[str, float | bool]:
        """Baseline knob values (repro bundle edits ``src/main.py`` module constants)."""
        return {
            "w_survival": float(sim_main.SEMANTIC_W_SURVIVAL),
            "w_pred_rss": float(sim_main.SEMANTIC_W_PREDICTED_RSS),
            "w_load": float(sim_main.SEMANTIC_W_LOAD),
            "w_aoi": float(sim_main.SEMANTIC_W_AOI),
            "enable_predictive_trigger": bool(sim_main.ENABLE_PREDICTIVE_TRIGGER),
            "enable_relay_fallback": bool(sim_main.ENABLE_RELAY_FALLBACK),
        }

    def _merged_policy_profile(self, algorithm_key: str) -> Dict[str, float | bool]:
        profile = dict(self._default_policy_profile_from_module())
        ov = self._ablation_profile(algorithm_key)
        if ov:
            if "w_survival" in ov:
                profile["w_survival"] = float(ov["w_survival"])
            if "w_aoi" in ov:
                profile["w_aoi"] = float(ov["w_aoi"])
            if "w_load" in ov:
                profile["w_load"] = float(ov["w_load"])
            if "w_pred_rss" in ov:
                profile["w_pred_rss"] = float(ov["w_pred_rss"])
            if "enable_predictive_trigger" in ov:
                profile["enable_predictive_trigger"] = bool(ov["enable_predictive_trigger"])
            if "enable_relay_fallback" in ov:
                profile["enable_relay_fallback"] = bool(ov["enable_relay_fallback"])
        return profile

    def run_single_trial(
        self,
        trial_id: int,
        duration_s: float = 30.0,
        trajectory_pattern: str = "random_walk",
        algorithm: str = "my_algorithm",
        seed: int = None,
        debug_mode: bool = False,
        scenario: Optional[str] = None,
        weather_mode: Optional[str] = None,
        packet_profile: Optional[str] = None,
        sim: Optional[V2XSmartHandoffSimulator] = None,
        policy_weight_override: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Run one complete simulation trial.

        Returns dict with metrics:
        - avg_aoi, peak_aoi, handoff_count, ping_pong_count, outage_time
        - time_series data for plotting
        """
        algorithm_key = self._normalize_algorithm(algorithm)
        self._validate_algorithm(algorithm_key)
        owns_sim = sim is None
        if owns_sim:
            sim = V2XSmartHandoffSimulator(headless=CONFIG.headless_monte_carlo)

        sim.enable_small_scale_fading_override = CONFIG.monte_carlo_enable_small_scale_fading

        scenario_name = scenario if scenario is not None else self.base_scenario
        sim.apply_scenario(scenario_name)
        apply_sim_environment(sim, weather_mode=weather_mode, packet_profile=packet_profile)

        merged = self._merged_policy_profile(algorithm_key)
        if policy_weight_override:
            for key, val in policy_weight_override.items():
                if key in merged:
                    merged[key] = float(val)
        sim.apply_policy_profile(merged)

        sim.debug_handoff = debug_mode
        pre_debug_m, pre_debug_h = sim.handoff_margin_db, sim.handoff_hold_s

        aoi_values: List[float] = []
        connected_rss_trace: List[float] = []
        rss_values = {tower.name: [] for tower in sim.towers}
        handoff_events: List[Dict] = []
        connected_towers: List[str] = []
        outage_frames = 0

        try:
            if debug_mode:
                sim.handoff_margin_db = 2.0
                sim.handoff_hold_s = 0.20

            self.randomize_scenario_parameters(sim, seed=seed)
            policy_state = self._init_policy_state(sim, algorithm_key)

            if trajectory_pattern != "mouse":
                positions = self.generate_synthetic_trajectory(duration_s, trajectory_pattern)
                sim.replay_positions = positions
                sim.replay_mode = True
                sim.replay_index = 0
            else:
                sim.replay_mode = False
                sim.replay_positions = []
                sim.replay_index = 0

            dt = 1.0 / FPS
            n_frames = int(duration_s * FPS)

            sim.reset_monte_carlo_trial_state()

            for frame in range(n_frames):
                sim.update_ue_kinematics(dt)
                if algorithm_key in {
                    "my_algorithm",
                    "ab_no_survival",
                    "ab_no_aoi",
                    "ab_no_load",
                    "ab_reactive_only",
                    "ab_no_relay",
                }:
                    measurements = sim.update_handoff_logic()
                else:
                    measurements = {t: sim.calculate_measurement(t, sim.ue_pos, sim.ue_velocity) for t in sim.towers}
                    self._apply_baseline_handoff(sim, measurements, algorithm_key, dt, policy_state)
                sim.update_aoi_state(measurements, dt)

                aoi_values.append(sim.aoi_age_s)
                connected_towers.append(sim.current_tower.name)

                for tower in sim.towers:
                    rss_values[tower.name].append(measurements[tower].rss_dbm)

                if sim.connection_mode == "tower":
                    current_rss = measurements[sim.current_tower].rss_dbm
                else:
                    current_rss = measurements[sim.current_tower].rss_dbm - sim_main.RELAY_PENALTY_DB
                connected_rss_trace.append(current_rss)
                if current_rss < OUTAGE_RSS_THRESHOLD_DBM:
                    outage_frames += 1

                if frame > 0 and connected_towers[-1] != connected_towers[-2]:
                    handoff_events.append(
                        {
                            "frame": frame,
                            "time": frame * dt,
                            "from": connected_towers[-2],
                            "to": connected_towers[-1],
                        }
                    )

                if sim.replay_mode and sim.replay_index >= len(sim.replay_positions):
                    break
        finally:
            sim.handoff_margin_db, sim.handoff_hold_s = pre_debug_m, pre_debug_h
            if owns_sim:
                import pygame

                pygame.quit()

        handoff_count = len(handoff_events)

        ping_pong_count = 0
        for i in range(2, len(connected_towers)):
            if connected_towers[i] == connected_towers[i - 2] and connected_towers[i] != connected_towers[i - 1]:
                ping_pong_count += 1
        ping_pong_count = ping_pong_count // 2

        outage_time = outage_frames * dt
        avg_aoi = float(np.mean(aoi_values)) if aoi_values else 0.0
        peak_aoi = float(np.max(aoi_values)) if aoi_values else 0.0
        connected_rss_values = [
            rss_values[tower_name][i]
            for i, tower_name in enumerate(connected_towers)
            if i < len(rss_values[tower_name])
        ]
        avg_rss = float(np.mean(connected_rss_values)) if connected_rss_values else 0.0

        trial_out = {
            "trial_id": trial_id,
            "algorithm": algorithm,
            "trajectory_pattern": trajectory_pattern,
            "experiment_scenario": scenario_name,
            "weather_mode": sim.weather_mode,
            "packet_profile": sim.packet_profile,
            "duration_s": duration_s,
            "avg_aoi": avg_aoi,
            "peak_aoi": peak_aoi,
            "handoff_count": handoff_count,
            "ping_pong_count": ping_pong_count,
            "outage_time_s": outage_time,
            "avg_rss_dbm": avg_rss,
            "aoi_values": aoi_values,
            "connected_towers": connected_towers,
            "rss_values": rss_values,
            "connected_rss_trace": connected_rss_trace,
            "handoff_events": handoff_events,
            "seed": seed,
        }
        trial_out["avg_aoi_external"] = compute_external_aoi(trial_out)
        trial_out["peak_aoi_external"] = compute_external_aoi_peak(trial_out)
        return trial_out

    def _validate_algorithm(self, algorithm_key: str) -> None:
        valid = {
            "my_algorithm",
            "a3_3gpp",
            "a3_robust",
            "greedy_rss",
            "q_learning",
            "mpc_lookahead",
            "ab_no_survival",
            "ab_no_aoi",
            "ab_no_load",
            "ab_reactive_only",
            "ab_no_relay",
        }
        if algorithm_key not in valid:
            raise ValueError(f"Unsupported algorithm '{algorithm_key}'. Supported: {sorted(valid)}")

    def _init_policy_state(self, sim: V2XSmartHandoffSimulator, algorithm_key: str) -> Dict:
        state: Dict[str, object] = {
            "candidate_tower": None,
            "candidate_since": 0.0,
        }
        if algorithm_key == "q_learning":
            state["agent"] = QLearningHandoff(sim.towers)
            state["last_state"] = None
            state["last_action"] = None
        if algorithm_key == "a3_robust":
            state["robust_a3"] = RobustA3Baseline()
        if algorithm_key == "mpc_lookahead":
            state["mpc"] = MPCLookaheadBaseline()
        return state

    def _execute_handoff(
        self,
        sim: V2XSmartHandoffSimulator,
        target_tower: Tower,
        reason: str,
    ) -> None:
        if target_tower == sim.current_tower:
            return
        old_name = sim.current_tower.name
        sim.current_tower = target_tower
        sim.handoff_spike_until = sim.sim_time_s + sim_main.HANDOFF_SPIKE_S
        sim.candidate_tower = None
        sim.push_event(f"{reason} HANDOFF {old_name} -> {sim.current_tower.name}")

    def _apply_baseline_handoff(
        self,
        sim: V2XSmartHandoffSimulator,
        measurements: Dict[Tower, Measurement],
        algorithm_key: str,
        dt: float,
        policy_state: Dict,
    ) -> None:
        strongest = max(sim.towers, key=lambda t: measurements[t].rss_dbm)
        current_rss = measurements[sim.current_tower].rss_dbm
        strongest_rss = measurements[strongest].rss_dbm

        if algorithm_key == "greedy_rss":
            if strongest != sim.current_tower:
                self._execute_handoff(sim, strongest, "GREEDY")
            return

        if algorithm_key == "a3_3gpp":
            hysteresis_db = 3.0
            ttt_s = 0.5
            cand = policy_state["candidate_tower"]
            if strongest != sim.current_tower and strongest_rss > current_rss + hysteresis_db:
                if cand != strongest:
                    policy_state["candidate_tower"] = strongest
                    policy_state["candidate_since"] = sim.sim_time_s
                    sim.candidate_tower = strongest
                elif sim.sim_time_s - float(policy_state["candidate_since"]) >= ttt_s:
                    self._execute_handoff(sim, strongest, "A3")
                    policy_state["candidate_tower"] = None
            else:
                policy_state["candidate_tower"] = None
                sim.candidate_tower = None
            return

        if algorithm_key == "a3_robust":
            baseline: RobustA3Baseline = policy_state["robust_a3"]
            chosen, did, _ = baseline.decide(sim, measurements)
            if did:
                self._execute_handoff(sim, chosen, "ROBUST_A3")
            return

        if algorithm_key == "mpc_lookahead":
            mpc: MPCLookaheadBaseline = policy_state["mpc"]
            chosen, did, _ = mpc.decide(sim, measurements)
            if did:
                self._execute_handoff(sim, chosen, "MPC")
            return

        if algorithm_key == "q_learning":
            agent: QLearningHandoff = policy_state["agent"]
            speed_mps = (sim.ue_velocity[0] ** 2 + sim.ue_velocity[1] ** 2) ** 0.5
            serving = sim.current_tower
            serving_m = measurements[serving]
            state = agent.discretize(serving_m.rss_dbm, speed_mps, serving_m.distance_m)
            action = agent.choose_action(state)

            # AoI-aligned reward for fairer comparison with AoI-centric metrics.
            reward = -sim.aoi_age_s
            reward -= 0.15 if action != serving else 0.0
            reward -= 0.35 if measurements[action].rss_dbm < OUTAGE_RSS_THRESHOLD_DBM else 0.0

            if action != serving:
                self._execute_handoff(sim, action, "QLEARN")

            next_serving = sim.current_tower
            next_m = measurements[next_serving]
            next_state = agent.discretize(next_m.rss_dbm, speed_mps, next_m.distance_m)
            agent.update(state, action, reward, next_state)
            return

    def run_monte_carlo(
        self,
        n_trials: int = 100,
        duration_s: float = 30.0,
        trajectory_pattern: str = "random_walk",
        algorithm: str = "my_algorithm",
        base_seed: int = 42,
        debug_mode: bool = False,
        scenario: Optional[str] = None,
        weather_mode: Optional[str] = None,
        packet_profile: Optional[str] = None,
        policy_weight_override: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Run Monte Carlo simulation with n_trials.

        Returns aggregated statistics with confidence intervals.
        """
        from scipy import stats

        all_results = []

        env_bits = []
        if scenario is not None:
            env_bits.append(f"scenario={scenario}")
        if weather_mode is not None:
            env_bits.append(f"weather={weather_mode}")
        if packet_profile is not None:
            env_bits.append(f"profile={packet_profile}")
        env_str = (" | " + ", ".join(env_bits)) if env_bits else ""
        print(f"Running Monte Carlo: {n_trials} trials, pattern={trajectory_pattern}, algo={algorithm}{env_str}")
        print("-" * 70)

        import pygame

        sim = V2XSmartHandoffSimulator(headless=CONFIG.headless_monte_carlo)
        try:
            for i in range(n_trials):
                seed = base_seed + i if base_seed is not None else None
                result = self.run_single_trial(
                    trial_id=i,
                    duration_s=duration_s,
                    trajectory_pattern=trajectory_pattern,
                    algorithm=algorithm,
                    seed=seed,
                    debug_mode=debug_mode,
                    scenario=scenario,
                    weather_mode=weather_mode,
                    packet_profile=packet_profile,
                    sim=sim,
                    policy_weight_override=policy_weight_override,
                )
                all_results.append(result)

                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{n_trials} trials...")
        finally:
            pygame.quit()

        avg_aoi_values = [r["avg_aoi"] for r in all_results]
        peak_aoi_values = [r["peak_aoi"] for r in all_results]
        avg_aoi_ext_values = [r["avg_aoi_external"] for r in all_results]
        peak_aoi_ext_values = [r["peak_aoi_external"] for r in all_results]
        handoff_values = [r["handoff_count"] for r in all_results]
        ping_pong_values = [r["ping_pong_count"] for r in all_results]
        outage_values = [r["outage_time_s"] for r in all_results]

        def compute_ci(data, confidence=0.95):
            n = len(data)
            mean = float(np.mean(data))
            if n < 2:
                return mean, (mean, mean), 0.0
            sem = stats.sem(data)
            if not np.isfinite(sem) or sem == 0.0:
                return mean, (mean, mean), float(np.std(data))
            ci = stats.t.interval(confidence, n - 1, loc=mean, scale=sem)
            return mean, ci, float(np.std(data))

        stats_dict = {
            "algorithm": algorithm,
            "trajectory_pattern": trajectory_pattern,
            "n_trials": n_trials,
            "duration_per_trial_s": duration_s,
            "experiment_scenario": all_results[0]["experiment_scenario"] if all_results else None,
            "weather_mode": all_results[0]["weather_mode"] if all_results else None,
            "packet_profile": all_results[0]["packet_profile"] if all_results else None,
        }

        for metric_name, metric_values in [
            ("avg_aoi", avg_aoi_values),
            ("peak_aoi", peak_aoi_values),
            ("avg_aoi_external", avg_aoi_ext_values),
            ("peak_aoi_external", peak_aoi_ext_values),
            ("handoff_count", handoff_values),
            ("ping_pong_count", ping_pong_values),
            ("outage_time_s", outage_values),
        ]:
            mean, ci, std = compute_ci(metric_values)
            stats_dict[f"{metric_name}_mean"] = mean
            stats_dict[f"{metric_name}_std"] = std
            stats_dict[f"{metric_name}_ci_low"] = ci[0]
            stats_dict[f"{metric_name}_ci_high"] = ci[1]

        stats_dict["all_results"] = all_results
        self.results_history.append(stats_dict)

        print("\n" + "=" * 70)
        print(f"MONTE CARLO RESULTS: {algorithm} ({trajectory_pattern})")
        print("=" * 70)
        print(f"Trials: {n_trials} x {duration_s}s = {n_trials * duration_s}s total")
        print(f"\nAverage AoI:     {stats_dict['avg_aoi_mean']:.4f} +/- {stats_dict['avg_aoi_std']:.4f} s")
        print(f"                 95% CI: [{stats_dict['avg_aoi_ci_low']:.4f}, {stats_dict['avg_aoi_ci_high']:.4f}]")
        print(f"\nPeak AoI:        {stats_dict['peak_aoi_mean']:.4f} +/- {stats_dict['peak_aoi_std']:.4f} s")
        print(f"Avg AoI (ext.):  {stats_dict['avg_aoi_external_mean']:.4f} +/- {stats_dict['avg_aoi_external_std']:.4f} s")
        print(f"Peak AoI (ext.): {stats_dict['peak_aoi_external_mean']:.4f} +/- {stats_dict['peak_aoi_external_std']:.4f} s")
        print(f"Handoffs:        {stats_dict['handoff_count_mean']:.2f} +/- {stats_dict['handoff_count_std']:.2f}")
        print(f"Ping-Pongs:      {stats_dict['ping_pong_count_mean']:.2f} +/- {stats_dict['ping_pong_count_std']:.2f}")
        print(f"Outage Time:     {stats_dict['outage_time_s_mean']:.3f} +/- {stats_dict['outage_time_s_std']:.3f} s")
        print("=" * 70 + "\n")

        return stats_dict

    def save_results_to_csv(self, stats_dict: Dict, output_dir: str = "outputs") -> Path:
        """Save Monte Carlo results to CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = _monte_carlo_csv_filename_slug(stats_dict)
        safe_algo = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in slug)
        safe_pattern = "".join(
            ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stats_dict["trajectory_pattern"]
        )
        extra = ""
        if stats_dict.get("experiment_scenario"):
            extra += "_" + "".join(
                ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(stats_dict["experiment_scenario"])
            )
        if stats_dict.get("weather_mode"):
            extra += "_" + str(stats_dict["weather_mode"])
        if stats_dict.get("packet_profile"):
            extra += "_" + str(stats_dict["packet_profile"])
        filename = f"monte_carlo_{safe_algo}_{safe_pattern}{extra}_{timestamp}.csv"
        filepath = output_dir / filename

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Algorithm", _monte_carlo_csv_algorithm_label(stats_dict)])
            writer.writerow(["Trajectory Pattern", stats_dict["trajectory_pattern"]])
            if stats_dict.get("experiment_scenario") is not None:
                writer.writerow(["Experiment Scenario", stats_dict["experiment_scenario"]])
            if stats_dict.get("weather_mode") is not None:
                writer.writerow(["Weather Mode", stats_dict["weather_mode"]])
            if stats_dict.get("packet_profile") is not None:
                writer.writerow(["Packet Profile", stats_dict["packet_profile"]])
            writer.writerow(["Number of Trials", stats_dict["n_trials"]])
            writer.writerow(["Duration per Trial (s)", stats_dict["duration_per_trial_s"]])
            writer.writerow([])
            writer.writerow(["Metric", "Mean", "Std Dev", "95% CI Low", "95% CI High"])

            for metric in [
                "avg_aoi",
                "peak_aoi",
                "avg_aoi_external",
                "peak_aoi_external",
                "handoff_count",
                "ping_pong_count",
                "outage_time_s",
            ]:
                writer.writerow(
                    [
                        metric,
                        f"{stats_dict[f'{metric}_mean']:.6f}",
                        f"{stats_dict[f'{metric}_std']:.6f}",
                        f"{stats_dict[f'{metric}_ci_low']:.6f}",
                        f"{stats_dict[f'{metric}_ci_high']:.6f}",
                    ]
                )

        return filepath


if __name__ == "__main__":
    evaluator = MonteCarloEvaluator(base_scenario="urban_canyon")

    results = evaluator.run_monte_carlo(
        n_trials=CONFIG.monte_carlo_trials,
        duration_s=CONFIG.monte_carlo_duration_s,
        trajectory_pattern="random_walk",
        algorithm="my_algorithm",
        base_seed=CONFIG.monte_carlo_base_seed,
    )

    csv_path = evaluator.save_results_to_csv(results)
    print(f"Results saved to: {csv_path}")
