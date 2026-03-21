import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from main import FPS, METER_PER_PIXEL, OUTAGE_RSS_THRESHOLD_DBM, Measurement, Tower, V2XSmartHandoffSimulator


class A3HandoffBaseline:
    """
    3GPP A3 Event-based Handoff (Industry Standard).

    Condition (simplified): neighbor_RSS > current_RSS + hysteresis
    """

    def __init__(self, hysteresis_db: float = 3.0, time_to_trigger_s: float = 0.5):
        self.hysteresis_db = hysteresis_db
        self.time_to_trigger_s = time_to_trigger_s
        self.candidate_tower: Optional[Tower] = None
        self.candidate_start_time: Optional[float] = None

    def reset(self) -> None:
        self.candidate_tower = None
        self.candidate_start_time = None

    def decide(
        self,
        sim: V2XSmartHandoffSimulator,
        measurements: Dict[Tower, Measurement],
    ) -> Tuple[Tower, bool, Optional[str]]:
        current_tower = sim.current_tower
        current_rss = measurements[current_tower].rss_dbm
        strongest = max(sim.towers, key=lambda t: measurements[t].rss_dbm)
        strongest_rss = measurements[strongest].rss_dbm

        now_s = getattr(sim, "sim_time_s", None)
        if now_s is None:
            now_s = time.time()

        if strongest != current_tower:
            if strongest_rss > current_rss + self.hysteresis_db:
                if self.candidate_tower != strongest:
                    self.candidate_tower = strongest
                    self.candidate_start_time = now_s
                elif self.candidate_start_time is not None and (now_s - self.candidate_start_time >= self.time_to_trigger_s):
                    old_name = current_tower.name
                    self.reset()
                    return strongest, True, f"A3 HANDOFF {old_name} -> {strongest.name}"
            else:
                if self.candidate_tower == strongest:
                    self.reset()

        return current_tower, False, None


class RobustA3Baseline:
    """
    A3-style event with load-aware effective RSS and time-to-trigger (TTT).

    Candidate attractiveness: ``RSS - load_offset_db * load`` (penalizes congested cells).
    Entry condition (vs serving): ``effective_candidate > serving_RSS + hysteresis_db``,
    then TTT must elapse before handoff (same structure as :class:`A3HandoffBaseline`).
    """

    def __init__(
        self,
        hysteresis_db: float = 3.0,
        time_to_trigger_s: float = 0.5,
        load_offset_db: float = 2.0,
    ):
        self.hysteresis_db = hysteresis_db
        self.time_to_trigger_s = time_to_trigger_s
        self.load_offset_db = load_offset_db
        self.candidate_tower: Optional[Tower] = None
        self.candidate_start_time: Optional[float] = None

    def reset(self) -> None:
        self.candidate_tower = None
        self.candidate_start_time = None

    def _effective_rss(
        self,
        sim: V2XSmartHandoffSimulator,
        tower: Tower,
        measurements: Dict[Tower, Measurement],
    ) -> float:
        rss = measurements[tower].rss_dbm
        load = float(sim.tower_loads.get(tower.name, 0.5))
        return rss - self.load_offset_db * load

    def decide(
        self,
        sim: V2XSmartHandoffSimulator,
        measurements: Dict[Tower, Measurement],
    ) -> Tuple[Tower, bool, Optional[str]]:
        current_tower = sim.current_tower
        current_rss = measurements[current_tower].rss_dbm

        others = [t for t in sim.towers if t != current_tower]
        if not others:
            return current_tower, False, None

        best = max(others, key=lambda t: self._effective_rss(sim, t, measurements))
        best_effective = self._effective_rss(sim, best, measurements)

        now_s = getattr(sim, "sim_time_s", None)
        if now_s is None:
            now_s = time.time()

        if best_effective > current_rss + self.hysteresis_db:
            if self.candidate_tower != best:
                self.candidate_tower = best
                self.candidate_start_time = now_s
            elif self.candidate_start_time is not None and (
                now_s - self.candidate_start_time >= self.time_to_trigger_s
            ):
                old_name = current_tower.name
                self.reset()
                return best, True, f"RobustA3 HANDOFF {old_name} -> {best.name}"
        else:
            if self.candidate_tower == best:
                self.reset()

        return current_tower, False, None


class GreedyRSSBaseline:
    """Always connect to strongest instantaneous RSS."""

    def decide(
        self,
        sim: V2XSmartHandoffSimulator,
        measurements: Dict[Tower, Measurement],
    ) -> Tuple[Tower, bool, Optional[str]]:
        current_tower = sim.current_tower
        strongest = max(sim.towers, key=lambda t: measurements[t].rss_dbm)
        if strongest != current_tower:
            old_name = current_tower.name
            return strongest, True, f"GREEDY HANDOFF {old_name} -> {strongest.name}"
        return current_tower, False, None


class MPCLookaheadBaseline:
    """
    Short-horizon model-predictive style baseline: maximize sum of predicted RSS over
    ``horizon_s`` with step ``dt_s``, minus a one-time dB penalty if switching away
    from the serving cell at the first step (reduces ping-pong vs greedy).
    """

    def __init__(self, horizon_s: float = 1.0, dt_s: float = 0.1, handoff_cost_db: float = 3.0):
        self.horizon_s = float(horizon_s)
        self.dt_s = max(1e-6, float(dt_s))
        self.handoff_cost_db = float(handoff_cost_db)

    def decide(
        self,
        sim: V2XSmartHandoffSimulator,
        measurements: Dict[Tower, Measurement],
    ) -> Tuple[Tower, bool, Optional[str]]:
        best_score = -np.inf
        best_tower = sim.current_tower
        current_tower = sim.current_tower

        n_steps = max(1, int(np.ceil(self.horizon_s / self.dt_s)))

        for candidate in sim.towers:
            score = 0.0
            for step in range(n_steps):
                lookahead = min(step * self.dt_s, self.horizon_s)
                pred_pos = sim.get_predicted_position(lookahead)
                pos = (float(pred_pos[0]), float(pred_pos[1]))
                m = sim.calculate_measurement(
                    candidate,
                    pos,
                    sim.ue_velocity,
                    future_s=lookahead,
                )
                pred_rss = m.rss_dbm
                penalty = (
                    self.handoff_cost_db
                    if (candidate != current_tower and step == 0)
                    else 0.0
                )
                score += pred_rss - penalty

            if score > best_score:
                best_score = score
                best_tower = candidate

        if best_tower != current_tower:
            old_name = current_tower.name
            return best_tower, True, f"MPC HANDOFF {old_name} -> {best_tower.name}"
        return current_tower, False, None


class QLearningHandoffBaseline:
    """
    Simple Q-learning baseline.

    State: (current_RSS_bin, speed_bin, dist_to_strongest_bin)
    Action: choose tower
    Reward: -AoI with light handoff/outage penalties
    """

    def __init__(self, towers: List[Tower], alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1):
        self.towers = towers
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: Dict[Tuple[int, int, int], Dict[str, float]] = {}
        self.last_state: Optional[Tuple[int, int, int]] = None
        self.last_action: Optional[Tower] = None
        self.last_reward: Optional[float] = None

    def _discretize_state(
        self,
        sim: V2XSmartHandoffSimulator,
        measurements: Dict[Tower, Measurement],
    ) -> Tuple[int, int, int]:
        current_rss = measurements[sim.current_tower].rss_dbm
        speed = math.hypot(sim.ue_velocity[0], sim.ue_velocity[1])
        strongest = max(sim.towers, key=lambda t: measurements[t].rss_dbm)
        dist_to_strongest = measurements[strongest].distance_m
        rss_bin = int((current_rss + 140.0) // 10.0)
        speed_bin = int(speed // 5.0)
        dist_bin = int(dist_to_strongest // 50.0)
        return (rss_bin, speed_bin, dist_bin)

    def _get_q_values(self, state: Tuple[int, int, int]) -> Dict[str, float]:
        if state not in self.q_table:
            self.q_table[state] = {tower.name: 0.0 for tower in self.towers}
        return self.q_table[state]

    def choose_action(
        self,
        sim: V2XSmartHandoffSimulator,
        measurements: Dict[Tower, Measurement],
    ) -> Tower:
        state = self._discretize_state(sim, measurements)
        if random.random() < self.epsilon:
            return random.choice(self.towers)
        q_values = self._get_q_values(state)
        best_tower_name = max(q_values, key=q_values.get)
        return next(t for t in self.towers if t.name == best_tower_name)

    def update(
        self,
        sim: V2XSmartHandoffSimulator,
        measurements: Dict[Tower, Measurement],
        reward: float,
        done: bool = False,
    ) -> None:
        state = self._discretize_state(sim, measurements)
        if self.last_state is not None and self.last_action is not None and self.last_reward is not None:
            old_value = self._get_q_values(self.last_state)[self.last_action.name]
            max_next_q = 0.0 if done else max(self._get_q_values(state).values())
            new_value = old_value + self.alpha * (self.last_reward + self.gamma * max_next_q - old_value)
            self.q_table[self.last_state][self.last_action.name] = new_value
        if not done:
            self.last_state = state
            self.last_action = sim.current_tower
            self.last_reward = reward

    def decide(
        self,
        sim: V2XSmartHandoffSimulator,
        measurements: Dict[Tower, Measurement],
    ) -> Tuple[Tower, bool, Optional[str]]:
        current_tower = sim.current_tower
        chosen_tower = self.choose_action(sim, measurements)
        if chosen_tower != current_tower:
            old_name = current_tower.name
            return chosen_tower, True, f"QL HANDOFF {old_name} -> {chosen_tower.name}"
        return current_tower, False, None


class BaselineRunner:
    """Run baseline algorithms on a fixed trajectory."""

    def __init__(self, sim: V2XSmartHandoffSimulator):
        self.sim = sim
        self.a3 = A3HandoffBaseline()
        self.robust_a3 = RobustA3Baseline()
        self.mpc = MPCLookaheadBaseline()
        self.greedy = GreedyRSSBaseline()
        self.qlearning: Optional[QLearningHandoffBaseline] = None

    def run_baseline_trial(self, algorithm: str, positions: List[Tuple[int, int]], duration_s: float = 30.0) -> Dict:
        sim = self.sim
        dt = 1.0 / FPS

        if algorithm == "qlearning":
            self.qlearning = QLearningHandoffBaseline(sim.towers)

        sim.ue_pos = positions[0]
        sim.prev_ue_pos = positions[0]
        sim.current_tower = sim.towers[0]
        sim.candidate_tower = None
        sim.aoi_age_s = 0.0

        aoi_values: List[float] = []
        handoff_count = 0
        outage_time = 0.0
        connected_towers: List[str] = []

        for idx, pos in enumerate(positions):
            sim.prev_ue_pos = sim.ue_pos
            sim.ue_pos = pos
            if idx > 0:
                vx = (pos[0] - positions[idx - 1][0]) * METER_PER_PIXEL / dt
                vy = (pos[1] - positions[idx - 1][1]) * METER_PER_PIXEL / dt
                sim.ue_velocity = (vx, vy)

            measurements = {t: sim.calculate_measurement(t, pos, sim.ue_velocity) for t in sim.towers}

            if algorithm == "a3":
                chosen_tower, handoff, _ = self.a3.decide(sim, measurements)
            elif algorithm == "robust_a3":
                chosen_tower, handoff, _ = self.robust_a3.decide(sim, measurements)
            elif algorithm == "greedy":
                chosen_tower, handoff, _ = self.greedy.decide(sim, measurements)
            elif algorithm == "mpc":
                chosen_tower, handoff, _ = self.mpc.decide(sim, measurements)
            elif algorithm == "qlearning":
                assert self.qlearning is not None
                reward = -sim.aoi_age_s
                self.qlearning.update(sim, measurements, reward, done=False)
                chosen_tower, handoff, _ = self.qlearning.decide(sim, measurements)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            if handoff:
                sim.current_tower = chosen_tower
                handoff_count += 1

            sim.update_aoi_state(measurements, dt)

            aoi_values.append(sim.aoi_age_s)
            connected_towers.append(sim.current_tower.name)
            current_rss = measurements[sim.current_tower].rss_dbm
            if current_rss < OUTAGE_RSS_THRESHOLD_DBM:
                outage_time += dt

        if algorithm == "qlearning" and self.qlearning is not None:
            self.qlearning.update(sim, measurements, -sim.aoi_age_s, done=True)

        return {
            "algorithm": algorithm,
            "avg_aoi": float(np.mean(aoi_values)) if aoi_values else 0.0,
            "peak_aoi": float(np.max(aoi_values)) if aoi_values else 0.0,
            "handoff_count": handoff_count,
            "outage_time_s": outage_time,
            "aoi_values": aoi_values,
            "connected_towers": connected_towers,
            "duration_s": duration_s,
        }
