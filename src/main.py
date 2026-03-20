import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

from config import CONFIG


WIDTH = 1280
HEIGHT = 800
FPS = 60

CITY_BG = (14, 18, 28)
GRID_COLOR = (33, 42, 60)
TEXT_COLOR = (230, 235, 245)
PANEL_BG = (8, 12, 20)
PANEL_BORDER = (60, 74, 96)
TOWER_COLORS = [(70, 186, 255), (255, 123, 91), (117, 230, 155), (190, 142, 255)]
BUILDING_COLOR = (130, 140, 160)
BUILDING_ALPHA = 130
BUILDING_BORDER = (170, 180, 200)
UE_COLOR = (245, 245, 250)
UE_RING = (255, 221, 107)
GOOD_COLOR = (100, 229, 146)
WARN_COLOR = (255, 205, 87)
BAD_COLOR = (255, 112, 112)

METER_PER_PIXEL = 0.1
FREQ_HZ = 5.9e9
LIGHT_SPEED = 299_792_458.0
PATH_LOSS_EXPONENT = 2.4
REFERENCE_DISTANCE_M = 1.0
REFERENCE_LOSS_DB = 32.44
TX_POWER_DBM = 30.0
SHADOWING_DROP_DB = 20.0
DYNAMIC_SHADOW_MEAN_DB = 14.0
DYNAMIC_SHADOW_STD_DB = 5.5
GHOST_NOISE_M = 1.8

HANDOFF_MARGIN_DB = 5.0
HANDOFF_HOLD_S = 0.5
HANDOFF_SPIKE_S = 0.035
PREDICT_LOOKAHEAD_S = 0.8
PREDICT_DT_S = 0.1
SURVIVAL_WINDOW_S = 1.0
SURVIVAL_DT_S = 0.1
OUTAGE_SLOPE_DB = 3.0
CRITICAL_AOI_DEADLINE_S = 0.010
STANDARD_AOI_DEADLINE_S = 1.0
RELAY_OUTAGE_GUARD_DBM = -92.0
RELAY_PENALTY_DB = 6.0
PACKET_SUCCESS_RSS_REF_DBM = -82.0
PACKET_SUCCESS_SLOPE_DB = 4.5
PACKET_SUCCESS_MIN_PROB = 0.02
PACKET_SUCCESS_MAX_PROB = 0.995
AOI_RESET_FLOOR_S = 0.008
TELEMETRY_SAMPLE_EVERY_N_FRAMES = 1
TELEMETRY_MAX_ROWS = 50_000

OUTPUT_DIR = Path("outputs")
OUTAGE_RSS_THRESHOLD_DBM = -95.0


@dataclass(frozen=True)
class Tower:
    name: str
    x: float
    y: float
    color: Tuple[int, int, int]


@dataclass
class Building:
    x: float
    y: float
    w: float
    h: float

    def normalized(self) -> "Building":
        nx = min(self.x, self.x + self.w)
        ny = min(self.y, self.y + self.h)
        nw = abs(self.w)
        nh = abs(self.h)
        return Building(nx, ny, nw, nh)

    def to_rect(self) -> pygame.Rect:
        n = self.normalized()
        return pygame.Rect(int(n.x), int(n.y), int(n.w), int(n.h))


@dataclass
class Measurement:
    distance_px: float
    distance_m: float
    rss_dbm: float
    doppler_hz: float
    blocked: bool
    risk_score: float = 0.0


@dataclass
class MovingBlocker:
    kind: str
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    base_loss_db: float
    jitter_db: float
    phase: float

    def predict_position(self, dt: float) -> Tuple[float, float]:
        return self.x + self.vx * dt, self.y + self.vy * dt


@dataclass
class RelayNode:
    name: str
    x: float
    y: float
    vx: float
    vy: float

    def predict_position(self, dt: float) -> Tuple[float, float]:
        return self.x + self.vx * dt, self.y + self.vy * dt


class Kalman2D:
    """Simple constant-velocity alpha-beta filter."""

    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.initialized = False
        self.alpha = 0.75
        self.beta = 0.2

    def update(self, mx: float, my: float, dt: float) -> None:
        if not self.initialized:
            self.x = mx
            self.y = my
            self.vx = 0.0
            self.vy = 0.0
            self.initialized = True
            return

        dt = max(1e-4, dt)
        x_pred = self.x + self.vx * dt
        y_pred = self.y + self.vy * dt
        rx = mx - x_pred
        ry = my - y_pred
        self.x = x_pred + self.alpha * rx
        self.y = y_pred + self.alpha * ry
        self.vx = self.vx + (self.beta / dt) * rx
        self.vy = self.vy + (self.beta / dt) * ry

    def predict(self, horizon_s: float) -> Tuple[float, float]:
        return self.x + self.vx * horizon_s, self.y + self.vy * horizon_s


def run_start_screen() -> bool:
    pygame.init()
    pygame.display.set_caption("V2X Smart Handoff Simulator - Start")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("segoe ui", 46, bold=True)
    subtitle_font = pygame.font.SysFont("consolas", 22)
    button_font = pygame.font.SysFont("segoe ui", 30, bold=True)

    button_w = 260
    button_h = 76
    button_rect = pygame.Rect((WIDTH - button_w) // 2, HEIGHT // 2 + 50, button_w, button_h)
    running = True
    should_start = False

    while running:
        mouse_pos = pygame.mouse.get_pos()
        hovered = button_rect.collidepoint(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE):
                should_start = True
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and hovered:
                should_start = True
                running = False

        screen.fill((9, 13, 20))

        title = title_font.render("Smart Handoff Simulator for V2X", True, TEXT_COLOR)
        subtitle = subtitle_font.render("Click Start to open the main simulator", True, (180, 194, 220))
        hint = subtitle_font.render("Tip: Press Enter or Space to start", True, (142, 166, 205))
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 2 - 160))
        screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, HEIGHT // 2 - 95))
        screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, HEIGHT // 2 - 60))

        button_color = (103, 167, 255) if hovered else (70, 132, 219)
        button_border = (195, 220, 255) if hovered else (140, 182, 242)
        pygame.draw.rect(screen, button_color, button_rect, border_radius=16)
        pygame.draw.rect(screen, button_border, button_rect, 2, border_radius=16)
        start_text = button_font.render("Start", True, (245, 249, 255))
        screen.blit(
            start_text,
            (
                button_rect.centerx - start_text.get_width() // 2,
                button_rect.centery - start_text.get_height() // 2,
            ),
        )

        pygame.display.flip()
        clock.tick(FPS)

    if not should_start:
        pygame.quit()
    return should_start


class V2XSmartHandoffSimulator:
    def __init__(self, headless: bool = False) -> None:
        self.headless = headless
        if self.headless and not os.environ.get("SDL_VIDEODRIVER"):
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        if self.headless:
            self.screen = pygame.Surface((WIDTH, HEIGHT))
        else:
            pygame.display.set_caption("V2X Smart Handoff Simulator - Enterprise")
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.start_ts = time.time()

        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 15)
        self.title_font = pygame.font.SysFont("segoe ui", 24, bold=True)

        # Keep towers inside the visible city canvas (outside right dashboard and top-left help zone).
        self.towers: List[Tower] = [
            Tower("TOWER_A", 140, 260, TOWER_COLORS[0]),
            Tower("TOWER_B", 760, 210, TOWER_COLORS[1]),
            Tower("TOWER_C", 330, HEIGHT - 170, TOWER_COLORS[2]),
            Tower("TOWER_D", 780, HEIGHT - 180, TOWER_COLORS[3]),
        ]
        self.buildings: List[Building] = []
        self.dynamic_blockers: List[MovingBlocker] = []
        self.relays: List[RelayNode] = []
        self.drag_building_start: Optional[Tuple[int, int]] = None

        self.ue_pos = (WIDTH // 2, HEIGHT // 2)
        self.prev_ue_pos = self.ue_pos
        self.ue_velocity = (0.0, 0.0)

        self.current_tower: Tower = self.towers[0]
        self.candidate_tower: Optional[Tower] = None
        self.candidate_since = 0.0
        self.sim_time_s = 0.0

        self.predictive_mode = "velocity"
        self.kalman = Kalman2D()

        self.events: List[str] = []
        self.max_events = 10
        self.show_help = True
        self.running = True

        self.telemetry: List[Dict[str, object]] = []
        self.last_export_path: Optional[Path] = None

        self.replay_mode = False
        self.replay_index = 0
        self.replay_positions: List[Tuple[int, int]] = []
        self.replay_source: Optional[Path] = None
        self.active_scenario = "baseline"
        self.last_risk_scores: Dict[str, float] = {}
        self.last_load_scores: Dict[str, float] = {}
        self.last_aoi_scores: Dict[str, float] = {}
        self.last_semantic_scores: Dict[str, float] = {}
        self.tower_loads: Dict[str, float] = {}
        self.connection_mode = "tower"
        self.relay_tower_name = ""
        self.relay_name = ""
        self.packet_profile = "critical"
        self.aoi_age_s = 0.0
        self.weather_mode = "clear"
        self.weather_entropy = 1.0
        self.handoff_spike_until = 0.0
        self.debug_handoff = False
        self.last_handoff_debug: Dict[str, object] = {}
        self.telemetry_frame_count = 0

        OUTPUT_DIR.mkdir(exist_ok=True)
        self.apply_scenario("baseline")

    def push_event(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        self.events.insert(0, entry)
        if len(self.events) > self.max_events:
            self.events.pop()

    def clamp_ue(self, x: int, y: int) -> Tuple[int, int]:
        return max(0, min(WIDTH - 1, x)), max(0, min(HEIGHT - 1, y))

    def get_dashboard_rect(self) -> pygame.Rect:
        return pygame.Rect(WIDTH - 430, 16, 414, HEIGHT - 32)

    def get_help_lines(self) -> List[str]:
        return [
            "Controls:",
            "- Move mouse: drive UE",
            "- Drag LMB: draw building",
            "- 1/2/3: load scenario preset",
            "- K: switch predictor (velocity/kalman)",
            "- U: packet urgency (critical/standard)",
            "- W: weather (clear/rain/fog/storm)",
            "- X: export telemetry CSV",
            "- P: generate performance chart",
            "- R: toggle replay from latest CSV",
            "- E: run side-by-side A/B evaluator",
            "- C: clear buildings | H: toggle help | ESC: quit",
        ]

    def get_help_rect(self) -> pygame.Rect:
        lines = len(self.get_help_lines())
        h = 24 + lines * 19
        return pygame.Rect(14, 14, 470, h)

    def blocked_ui_rects(self) -> List[pygame.Rect]:
        return [self.get_dashboard_rect(), self.get_help_rect()]

    def is_safe_point(self, x: int, y: int, radius: int = 18) -> bool:
        safe_rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
        for rect in self.blocked_ui_rects():
            if safe_rect.colliderect(rect):
                return False
        return True

    def snap_to_safe_point(self, x: int, y: int, radius: int = 18) -> Tuple[int, int]:
        sx, sy = self.clamp_ue(x, y)
        if self.is_safe_point(sx, sy, radius=radius):
            return sx, sy
        max_r = 420
        step = 10
        for r in range(step, max_r, step):
            for _ in range(32):
                theta = random.uniform(0, math.tau)
                nx = int(sx + r * math.cos(theta))
                ny = int(sy + r * math.sin(theta))
                nx, ny = self.clamp_ue(nx, ny)
                if self.is_safe_point(nx, ny, radius=radius):
                    return nx, ny
        return sx, sy

    def place_towers_for_scenario(self, name: str) -> None:
        old_current_name = self.current_tower.name if hasattr(self, "current_tower") else "TOWER_A"
        old_candidate_name = self.candidate_tower.name if self.candidate_tower else None
        templates = {
            "baseline": [(160, 240), (820, 220), (340, HEIGHT - 170), (800, HEIGHT - 180)],
            "urban_canyon": [(170, 170), (840, 170), (260, 640), (860, 620)],
            "intersection": [(210, 140), (770, 140), (250, 610), (770, 610)],
        }
        selected = templates.get(name, templates["baseline"])
        new_towers: List[Tower] = []
        for idx, (x, y) in enumerate(selected):
            sx, sy = self.snap_to_safe_point(x, y, radius=26)
            new_towers.append(Tower(f"TOWER_{chr(ord('A') + idx)}", sx, sy, TOWER_COLORS[idx]))
        self.towers = new_towers
        tower_by_name = {tower.name: tower for tower in self.towers}
        self.current_tower = tower_by_name.get(old_current_name, self.towers[0])
        self.candidate_tower = tower_by_name.get(old_candidate_name) if old_candidate_name else None

    def apply_scenario(self, name: str) -> None:
        self.active_scenario = name
        self.place_towers_for_scenario(name)
        if name == "baseline":
            self.buildings = []
            self.dynamic_blockers = [
                MovingBlocker("pedestrian", 320, 360, 18, 0, 12, 9.0, 2.4, 0.2),
                MovingBlocker("bus", 220, 600, 40, -8, 22, 13.5, 3.8, 1.1),
            ]
            self.relays = [
                RelayNode("RELAY_1", 500, 300, 10, 12),
                RelayNode("RELAY_2", 260, 480, 15, -10),
            ]
            self.ue_pos = self.snap_to_safe_point(WIDTH // 2, HEIGHT // 2, radius=14)
            self.prev_ue_pos = self.ue_pos
        elif name == "urban_canyon":
            self.buildings = [
                Building(460, 80, 110, 260),
                Building(640, 180, 130, 310),
                Building(500, 560, 260, 120),
            ]
            self.dynamic_blockers = [
                MovingBlocker("truck", 260, 450, 46, 0, 28, 16.0, 4.8, 0.0),
                MovingBlocker("pedestrian", 560, 160, 12, 22, 11, 8.0, 2.0, 1.6),
                MovingBlocker("bus", 770, 520, -34, -6, 22, 13.5, 3.2, 2.2),
            ]
            self.relays = [
                RelayNode("RELAY_1", 380, 580, 9, -15),
                RelayNode("RELAY_2", 700, 420, -12, 8),
            ]
            self.ue_pos = self.snap_to_safe_point(220, 520, radius=14)
            self.prev_ue_pos = self.ue_pos
        elif name == "intersection":
            self.buildings = [
                Building(420, 280, 170, 120),
                Building(690, 280, 170, 120),
                Building(550, 450, 160, 120),
            ]
            self.dynamic_blockers = [
                MovingBlocker("truck", 410, 640, 26, -34, 27, 15.0, 4.4, 0.7),
                MovingBlocker("bus", 860, 300, -48, 15, 22, 12.8, 3.6, 1.2),
                MovingBlocker("pedestrian", 600, 90, 0, 18, 10, 7.2, 2.1, 2.8),
            ]
            self.relays = [
                RelayNode("RELAY_1", 470, 330, 18, 0),
                RelayNode("RELAY_2", 820, 560, -16, -6),
            ]
            self.ue_pos = self.snap_to_safe_point(640, 680, radius=14)
            self.prev_ue_pos = self.ue_pos
        self.tower_loads = {tower.name: random.uniform(0.2, 0.6) for tower in self.towers}
        self.connection_mode = "tower"
        self.relay_name = ""
        self.relay_tower_name = ""
        self.push_event(f"Scenario loaded: {name}")

    def distance_point_to_segment(
        self, p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]
    ) -> float:
        ax, ay = a
        bx, by = b
        px, py = p
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        denom = abx * abx + aby * aby
        if denom <= 1e-9:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
        cx = ax + t * abx
        cy = ay + t * aby
        return math.hypot(px - cx, py - cy)

    def blocker_intersects_link(
        self, blocker: MovingBlocker, p1: Tuple[float, float], p2: Tuple[float, float], future_s: float = 0.0
    ) -> bool:
        bx, by = blocker.predict_position(future_s)
        # Ghost-blocker uncertainty: sensed blocker location has stochastic offset.
        if future_s > 0.0:
            bx += random.gauss(0.0, GHOST_NOISE_M / METER_PER_PIXEL)
            by += random.gauss(0.0, GHOST_NOISE_M / METER_PER_PIXEL)
        return self.distance_point_to_segment((bx, by), p1, p2) <= blocker.radius

    def blocker_shadow_drop(self, blocker: MovingBlocker, time_s: float) -> float:
        harmonic = math.sin(time_s * (0.8 + blocker.radius / 25.0) + blocker.phase)
        stochastic = random.gauss(0.0, blocker.jitter_db * self.weather_entropy)
        drop = blocker.base_loss_db + harmonic * blocker.jitter_db * 0.35 + stochastic * 0.25
        return max(4.0, drop)

    def update_dynamic_blockers(self, dt: float) -> None:
        if dt <= 0.0:
            return
        for blocker in self.dynamic_blockers:
            blocker.x += blocker.vx * dt
            blocker.y += blocker.vy * dt
            # Soft bounce to keep moving blockers in the city canvas.
            if blocker.x < 30 or blocker.x > WIDTH - 460:
                blocker.vx *= -1.0
                blocker.x = max(30.0, min(WIDTH - 460.0, blocker.x))
            if blocker.y < 30 or blocker.y > HEIGHT - 30:
                blocker.vy *= -1.0
                blocker.y = max(30.0, min(HEIGHT - 30.0, blocker.y))

    def update_relays(self, dt: float) -> None:
        if dt <= 0.0:
            return
        for relay in self.relays:
            relay.x += relay.vx * dt
            relay.y += relay.vy * dt
            if relay.x < 30 or relay.x > WIDTH - 460:
                relay.vx *= -1.0
                relay.x = max(30.0, min(WIDTH - 460.0, relay.x))
            if relay.y < 30 or relay.y > HEIGHT - 30:
                relay.vy *= -1.0
                relay.y = max(30.0, min(HEIGHT - 30.0, relay.y))

    def cycle_weather(self) -> None:
        order = ["clear", "rain", "fog", "storm"]
        idx = (order.index(self.weather_mode) + 1) % len(order)
        self.weather_mode = order[idx]
        profile = {"clear": 1.0, "rain": 1.3, "fog": 1.2, "storm": 1.8}
        self.weather_entropy = profile[self.weather_mode]
        self.push_event(f"Weather set: {self.weather_mode}")

    def toggle_packet_profile(self) -> None:
        self.packet_profile = "standard" if self.packet_profile == "critical" else "critical"
        self.push_event(f"Packet profile: {self.packet_profile}")

    def update_tower_loads(self, dt: float) -> None:
        for i, tower in enumerate(self.towers):
            now = self.sim_time_s
            base = self.tower_loads.get(tower.name, 0.4)
            drift = math.sin(now * 0.45 + i * 1.1) * 0.02
            noise = random.gauss(0.0, 0.015 * self.weather_entropy)
            nxt = max(0.05, min(0.98, base + drift + noise * max(0.3, dt * FPS / 60.0)))
            self.tower_loads[tower.name] = nxt

    def predict_tower_load(self, tower: Tower, horizon_s: float = PREDICT_LOOKAHEAD_S) -> float:
        now = self.sim_time_s
        idx = next(i for i, t in enumerate(self.towers) if t.name == tower.name)
        trend = math.sin((now + horizon_s) * 0.45 + idx * 1.1) * 0.04
        return max(0.05, min(0.99, self.tower_loads.get(tower.name, 0.4) + trend))

    def deadline_for_profile(self) -> float:
        return CRITICAL_AOI_DEADLINE_S if self.packet_profile == "critical" else STANDARD_AOI_DEADLINE_S

    def estimated_latency_s(self, tower: Tower, handoff: bool) -> float:
        load = self.predict_tower_load(tower)
        queue_delay = 0.012 + 0.11 * (load**2)
        weather_delay = 0.004 * (self.weather_entropy - 1.0)
        handoff_delay = HANDOFF_SPIKE_S if handoff else 0.0
        return queue_delay + weather_delay + handoff_delay

    def projected_aoi_ratio(self, tower: Tower, handoff: bool) -> float:
        projected = self.aoi_age_s + self.estimated_latency_s(tower, handoff)
        return projected / max(1e-4, self.deadline_for_profile())

    def evaluate_relay_fallback(self) -> None:
        current_rss = self.calculate_measurement(self.current_tower, self.ue_pos, self.ue_velocity).rss_dbm
        if current_rss > RELAY_OUTAGE_GUARD_DBM:
            self.connection_mode = "tower"
            self.relay_name = ""
            self.relay_tower_name = ""
            return
        best_path = None
        best_score = -1e9
        for relay in self.relays:
            ue_dist = max(1.0, math.hypot(self.ue_pos[0] - relay.x, self.ue_pos[1] - relay.y))
            ue_rss = TX_POWER_DBM - (REFERENCE_LOSS_DB + 10.0 * PATH_LOSS_EXPONENT * math.log10((ue_dist * METER_PER_PIXEL) / 1.0))
            relay_best = max(
                self.towers,
                key=lambda t: self.calculate_measurement(t, (relay.x, relay.y), (0.0, 0.0)).rss_dbm - self.predict_tower_load(t) * 12.0,
            )
            relay_tower_rss = self.calculate_measurement(relay_best, (relay.x, relay.y), (0.0, 0.0)).rss_dbm
            path_rss = min(ue_rss, relay_tower_rss) - RELAY_PENALTY_DB
            if path_rss > best_score:
                best_score = path_rss
                best_path = (relay, relay_best)
        if best_path and best_score > current_rss + 2.0:
            relay, relay_tower = best_path
            if self.connection_mode != "relay":
                self.push_event(f"V2V RELAY ENABLED via {relay.name}")
            self.connection_mode = "relay"
            self.relay_name = relay.name
            self.relay_tower_name = relay_tower.name
        else:
            if self.connection_mode == "relay":
                self.push_event("V2V RELAY disabled; tower path recovered.")
            self.connection_mode = "tower"
            self.relay_name = ""
            self.relay_tower_name = ""

    def load_replay_from_csv(self, path: Path) -> bool:
        positions: List[Tuple[int, int]] = []
        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    x = int(float(row["ue_x"]))
                    y = int(float(row["ue_y"]))
                    positions.append(self.clamp_ue(x, y))
        except (OSError, KeyError, ValueError):
            return False
        if len(positions) < 2:
            return False
        self.replay_positions = positions
        self.replay_index = 0
        self.replay_source = path
        return True

    def toggle_replay(self) -> None:
        if self.replay_mode:
            self.replay_mode = False
            self.push_event("Replay stopped.")
            return
        if self.last_export_path and self.load_replay_from_csv(self.last_export_path):
            self.replay_mode = True
            self.push_event(f"Replay started from {self.last_export_path.name}")
            return
        self.push_event("Replay unavailable. Export telemetry first (X).")

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                    self.place_towers_for_scenario(self.active_scenario)
                elif event.key == pygame.K_c:
                    self.buildings.clear()
                    self.push_event("All buildings removed.")
                elif event.key == pygame.K_1:
                    self.apply_scenario("baseline")
                elif event.key == pygame.K_2:
                    self.apply_scenario("urban_canyon")
                elif event.key == pygame.K_3:
                    self.apply_scenario("intersection")
                elif event.key == pygame.K_k:
                    self.predictive_mode = "kalman" if self.predictive_mode == "velocity" else "velocity"
                    self.push_event(f"Predictive model: {self.predictive_mode}")
                elif event.key == pygame.K_w:
                    self.cycle_weather()
                elif event.key == pygame.K_u:
                    self.toggle_packet_profile()
                elif event.key == pygame.K_x:
                    path = self.export_telemetry()
                    self.push_event(f"Telemetry exported: {path.name}")
                elif event.key == pygame.K_p:
                    chart = self.generate_performance_chart()
                    if chart:
                        self.push_event(f"Chart generated: {chart.name}")
                    else:
                        self.push_event("Chart generation skipped (need matplotlib + data).")
                elif event.key == pygame.K_r:
                    self.toggle_replay()
                elif event.key == pygame.K_e:
                    result = self.run_ab_evaluation()
                    if result:
                        self.push_event(f"A/B evaluation done: {result}")
                    else:
                        self.push_event("A/B evaluation skipped (need telemetry/replay trajectory).")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.replay_mode:
                self.drag_building_start = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and not self.replay_mode:
                if self.drag_building_start is not None:
                    sx, sy = self.drag_building_start
                    ex, ey = event.pos
                    b = Building(float(sx), float(sy), float(ex - sx), float(ey - sy)).normalized()
                    if b.w >= 14 and b.h >= 14:
                        self.buildings.append(b)
                        self.push_event(f"Building created at ({int(b.x)}, {int(b.y)}) size {int(b.w)}x{int(b.h)}")
                    self.drag_building_start = None

    def segments_intersect(
        self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float], d: Tuple[float, float]
    ) -> bool:
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(val) < 1e-9:
                return 0
            return 1 if val > 0 else 2

        def on_segment(p, q, r):
            return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

        o1 = orientation(a, b, c)
        o2 = orientation(a, b, d)
        o3 = orientation(c, d, a)
        o4 = orientation(c, d, b)
        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and on_segment(a, c, b):
            return True
        if o2 == 0 and on_segment(a, d, b):
            return True
        if o3 == 0 and on_segment(c, a, d):
            return True
        if o4 == 0 and on_segment(c, b, d):
            return True
        return False

    def line_intersects_rect(self, p1: Tuple[float, float], p2: Tuple[float, float], rect: Building) -> bool:
        r = rect.to_rect()
        if r.collidepoint(p1) or r.collidepoint(p2):
            return True
        corners = [(r.left, r.top), (r.right, r.top), (r.right, r.bottom), (r.left, r.bottom)]
        edges = [(corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[3]), (corners[3], corners[0])]
        return any(self.segments_intersect(p1, p2, a, b) for a, b in edges)

    def calculate_measurement(
        self,
        tower: Tower,
        ue_pos: Tuple[float, float],
        ue_vel: Tuple[float, float],
        future_s: float = 0.0,
    ) -> Measurement:
        dx = ue_pos[0] - tower.x
        dy = ue_pos[1] - tower.y
        distance_px = max(1.0, math.hypot(dx, dy))
        distance_m = max(0.5, distance_px * METER_PER_PIXEL)
        path_loss_db = REFERENCE_LOSS_DB + 10.0 * PATH_LOSS_EXPONENT * math.log10(distance_m / REFERENCE_DISTANCE_M)
        rss_dbm = TX_POWER_DBM - path_loss_db

        blocked = any(self.line_intersects_rect((tower.x, tower.y), ue_pos, b) for b in self.buildings)
        if blocked:
            rss_dbm -= SHADOWING_DROP_DB
        for blocker in self.dynamic_blockers:
            if self.blocker_intersects_link(blocker, (tower.x, tower.y), ue_pos, future_s=future_s):
                rss_dbm -= self.blocker_shadow_drop(blocker, self.sim_time_s + future_s)
                blocked = True
        if CONFIG.enable_small_scale_fading:
            if blocked:
                fading_amp = np.random.rayleigh(scale=1.0)
            else:
                k = max(0.0, CONFIG.rician_k_factor)
                fading_amp = math.sqrt(
                    np.random.normal(scale=math.sqrt(k / (k + 1.0))) ** 2
                    + np.random.normal(scale=math.sqrt(1.0 / (k + 1.0))) ** 2
                )
            rss_dbm += 20.0 * math.log10(max(1e-10, fading_amp))

        ux, uy = ue_vel
        speed = math.hypot(ux, uy)
        if speed < 1e-6:
            doppler_hz = 0.0
        else:
            radial = (dx / distance_px, dy / distance_px)
            radial_speed = ux * radial[0] + uy * radial[1]
            wavelength = LIGHT_SPEED / FREQ_HZ
            doppler_hz = radial_speed / wavelength

        return Measurement(distance_px, distance_m, rss_dbm, doppler_hz, blocked)

    def get_predicted_position(self, horizon_s: float) -> Tuple[int, int]:
        if self.predictive_mode == "kalman":
            px, py = self.kalman.predict(horizon_s)
        else:
            vx, vy = self.ue_velocity
            px = self.ue_pos[0] + (vx / METER_PER_PIXEL) * horizon_s
            py = self.ue_pos[1] + (vy / METER_PER_PIXEL) * horizon_s
        return self.clamp_ue(int(px), int(py))

    def predict_future_rss(self, tower: Tower, horizon_s: float = PREDICT_LOOKAHEAD_S) -> float:
        px, py = self.get_predicted_position(horizon_s)
        m = self.calculate_measurement(tower, (px, py), self.ue_velocity, future_s=horizon_s)
        return m.rss_dbm

    def outage_probability_from_rss(self, rss_dbm: float) -> float:
        x = (OUTAGE_RSS_THRESHOLD_DBM - rss_dbm) / max(0.2, OUTAGE_SLOPE_DB)
        return 1.0 / (1.0 + math.exp(-x))

    def packet_success_probability(self, rss_dbm: float) -> float:
        x = (rss_dbm - PACKET_SUCCESS_RSS_REF_DBM) / max(0.2, PACKET_SUCCESS_SLOPE_DB)
        p = 1.0 / (1.0 + math.exp(-x))
        return max(PACKET_SUCCESS_MIN_PROB, min(PACKET_SUCCESS_MAX_PROB, p))

    def evaluate_survival_score(self, tower: Tower, horizon_s: float = SURVIVAL_WINDOW_S) -> float:
        samples = max(1, int(horizon_s / SURVIVAL_DT_S))
        survival = 1.0
        rss_acc = 0.0
        for i in range(1, samples + 1):
            lookahead = i * SURVIVAL_DT_S
            px, py = self.get_predicted_position(lookahead)
            m = self.calculate_measurement(tower, (px, py), self.ue_velocity, future_s=lookahead)
            rss_acc += m.rss_dbm
            outage_prob = self.outage_probability_from_rss(m.rss_dbm)
            survival *= max(0.0, 1.0 - outage_prob * (SURVIVAL_DT_S / max(0.2, horizon_s)))
        avg_rss = rss_acc / samples
        rss_term = (avg_rss + 120.0) / 60.0
        return survival + 0.18 * rss_term

    def update_ue_kinematics(self, dt: float) -> None:
        if dt > 0.0:
            self.sim_time_s += dt
        self.update_dynamic_blockers(dt)
        self.update_relays(dt)
        self.update_tower_loads(dt)
        if self.replay_mode and self.replay_positions:
            self.prev_ue_pos = self.ue_pos
            self.ue_pos = self.replay_positions[self.replay_index]
            self.replay_index += 1
            if self.replay_index >= len(self.replay_positions):
                self.replay_index = 0
        else:
            mx, my = pygame.mouse.get_pos()
            mx, my = self.clamp_ue(mx, my)
            self.prev_ue_pos = self.ue_pos
            self.ue_pos = (mx, my)
        if dt > 1e-6:
            vx = (self.ue_pos[0] - self.prev_ue_pos[0]) * METER_PER_PIXEL / dt
            vy = (self.ue_pos[1] - self.prev_ue_pos[1]) * METER_PER_PIXEL / dt
            self.ue_velocity = (vx, vy)
        self.kalman.update(float(self.ue_pos[0]), float(self.ue_pos[1]), dt)

    def update_handoff_logic(self) -> Dict[Tower, Measurement]:
        measurements = {t: self.calculate_measurement(t, self.ue_pos, self.ue_velocity) for t in self.towers}
        strongest = max(self.towers, key=lambda t: measurements[t].rss_dbm)
        current_rss = measurements[self.current_tower].rss_dbm
        strongest_rss = measurements[strongest].rss_dbm

        predicted_current = self.predict_future_rss(self.current_tower)
        predicted_scores = {t: self.predict_future_rss(t) for t in self.towers}
        survival_scores = {t: self.evaluate_survival_score(t) for t in self.towers}
        load_scores = {t: self.predict_tower_load(t) for t in self.towers}
        aoi_scores = {t: self.projected_aoi_ratio(t, handoff=(t != self.current_tower)) for t in self.towers}
        semantic_scores = {
            t: survival_scores[t]
            + 0.22 * ((predicted_scores[t] + 110.0) / 45.0)
            - 0.60 * load_scores[t]
            - 0.80 * aoi_scores[t]
            for t in self.towers
        }
        self.last_risk_scores = {t.name: survival_scores[t] for t in self.towers}
        self.last_load_scores = {t.name: load_scores[t] for t in self.towers}
        self.last_aoi_scores = {t.name: aoi_scores[t] for t in self.towers}
        self.last_semantic_scores = {t.name: semantic_scores[t] for t in self.towers}
        predicted_best_tower = max(self.towers, key=lambda t: predicted_scores[t])
        survival_best_tower = max(self.towers, key=lambda t: survival_scores[t])
        semantic_best_tower = max(self.towers, key=lambda t: semantic_scores[t])
        adaptive_margin = HANDOFF_MARGIN_DB + (self.weather_entropy - 1.0) * 2.0
        predictive_pressure = predicted_scores[predicted_best_tower] - predicted_current >= HANDOFF_MARGIN_DB
        survival_pressure = (
            survival_best_tower != self.current_tower
            and survival_scores[survival_best_tower] - survival_scores[self.current_tower] >= 0.08
        )
        semantic_pressure = (
            semantic_best_tower != self.current_tower
            and semantic_scores[semantic_best_tower] - semantic_scores[self.current_tower] >= 0.06
        )

        trigger_by_margin = strongest != self.current_tower and (strongest_rss - current_rss >= adaptive_margin)
        trigger_predictive = predicted_best_tower != self.current_tower and predictive_pressure
        trigger_survival = survival_pressure
        trigger_semantic = semantic_pressure

        if trigger_by_margin:
            chosen_candidate = strongest
        elif trigger_semantic:
            chosen_candidate = semantic_best_tower
        elif trigger_survival:
            chosen_candidate = survival_best_tower
        else:
            chosen_candidate = predicted_best_tower
        should_consider = trigger_by_margin or trigger_predictive or trigger_survival or trigger_semantic
        if self.debug_handoff:
            self.last_handoff_debug = {
                "t": round(self.sim_time_s, 3),
                "current": self.current_tower.name,
                "strongest": strongest.name,
                "candidate": chosen_candidate.name if should_consider else "",
                "delta_rss_db": round(strongest_rss - current_rss, 3),
                "adaptive_margin_db": round(adaptive_margin, 3),
                "trigger_margin": int(trigger_by_margin),
                "trigger_predictive": int(trigger_predictive),
                "trigger_survival": int(trigger_survival),
                "trigger_semantic": int(trigger_semantic),
            }

        if should_consider:
            if self.candidate_tower != chosen_candidate:
                self.candidate_tower = chosen_candidate
                self.candidate_since = self.sim_time_s
            else:
                if self.sim_time_s - self.candidate_since >= HANDOFF_HOLD_S:
                    old_name = self.current_tower.name
                    self.current_tower = chosen_candidate
                    self.handoff_spike_until = self.sim_time_s + HANDOFF_SPIKE_S
                    self.candidate_tower = None
                    tag = "AOI " if trigger_semantic and not trigger_by_margin else ""
                    if trigger_survival and not trigger_by_margin and not trigger_semantic:
                        tag = "SURVIVAL "
                    if trigger_predictive and not trigger_by_margin and not trigger_survival:
                        tag = "PREDICTIVE "
                    self.push_event(f"{tag}HANDOFF {old_name} -> {self.current_tower.name}")
        else:
            self.candidate_tower = None
        self.evaluate_relay_fallback()
        return measurements

    def update_aoi_state(self, measurements: Dict[Tower, Measurement], dt: float) -> None:
        if dt <= 0:
            return
        if self.connection_mode == "tower":
            connected_rss = measurements[self.current_tower].rss_dbm
            connected_load = self.last_load_scores.get(self.current_tower.name, self.tower_loads.get(self.current_tower.name, 0.5))
        else:
            connected_rss = self.calculate_measurement(self.current_tower, self.ue_pos, self.ue_velocity).rss_dbm - RELAY_PENALTY_DB
            connected_load = self.last_load_scores.get(self.relay_tower_name, 0.35)
        freshness_gain = max(0.0, 0.90 - connected_load)
        outage_penalty = 0.0
        if connected_rss < OUTAGE_RSS_THRESHOLD_DBM:
            outage_penalty += dt * 4.0
        if self.sim_time_s <= self.handoff_spike_until:
            outage_penalty += dt * 2.0
        self.aoi_age_s = max(0.0, self.aoi_age_s + dt * (1.0 - freshness_gain) + outage_penalty)
        recv_prob = self.packet_success_probability(connected_rss) * max(0.05, 1.0 - connected_load)
        if random.random() < recv_prob:
            self.aoi_age_s = min(self.aoi_age_s, AOI_RESET_FLOOR_S)

    def record_telemetry(self, measurements: Dict[Tower, Measurement]) -> None:
        self.telemetry_frame_count += 1
        if TELEMETRY_SAMPLE_EVERY_N_FRAMES > 1 and (self.telemetry_frame_count % TELEMETRY_SAMPLE_EVERY_N_FRAMES != 0):
            return
        elapsed = self.sim_time_s
        if self.connection_mode == "tower":
            connected_rss = measurements[self.current_tower].rss_dbm
        else:
            connected_rss = (
                self.calculate_measurement(self.current_tower, self.ue_pos, self.ue_velocity).rss_dbm
                - RELAY_PENALTY_DB
            )
        row: Dict[str, object] = {
            "t": round(elapsed, 4),
            "ue_x": self.ue_pos[0],
            "ue_y": self.ue_pos[1],
            "ue_vx_mps": round(self.ue_velocity[0], 4),
            "ue_vy_mps": round(self.ue_velocity[1], 4),
            "ue_speed_mps": round(math.hypot(self.ue_velocity[0], self.ue_velocity[1]), 4),
            "current_tower": self.current_tower.name,
            "candidate_tower": self.candidate_tower.name if self.candidate_tower else "",
            "predictive_model": self.predictive_mode,
            "replay_mode": int(self.replay_mode),
            "weather_mode": self.weather_mode,
            "weather_entropy": round(self.weather_entropy, 4),
            "packet_profile": self.packet_profile,
            "aoi_age_s": round(self.aoi_age_s, 6),
            "aoi_deadline_s": round(self.deadline_for_profile(), 6),
            "connection_mode": self.connection_mode,
            "relay_name": self.relay_name,
            "relay_tower_name": self.relay_tower_name,
            "packet_success_prob": round(self.packet_success_probability(connected_rss), 6),
        }
        if self.last_handoff_debug:
            row.update({f"handoff_{k}": v for k, v in self.last_handoff_debug.items()})
        for tower in self.towers:
            row[f"{tower.name.lower()}_risk_score"] = round(self.last_risk_scores.get(tower.name, 0.0), 6)
            row[f"{tower.name.lower()}_load"] = round(self.last_load_scores.get(tower.name, 0.0), 6)
            row[f"{tower.name.lower()}_aoi_ratio"] = round(self.last_aoi_scores.get(tower.name, 0.0), 6)
            row[f"{tower.name.lower()}_semantic"] = round(self.last_semantic_scores.get(tower.name, 0.0), 6)
        for tower in self.towers:
            m = measurements[tower]
            key = tower.name.lower()
            row[f"{key}_rss_dbm"] = round(m.rss_dbm, 4)
            row[f"{key}_doppler_hz"] = round(m.doppler_hz, 4)
            row[f"{key}_distance_m"] = round(m.distance_m, 4)
            row[f"{key}_blocked"] = int(m.blocked)
        self.telemetry.append(row)
        if len(self.telemetry) > TELEMETRY_MAX_ROWS:
            del self.telemetry[: len(self.telemetry) - TELEMETRY_MAX_ROWS]

    def export_telemetry(self) -> Path:
        if not self.telemetry:
            self.push_event("No telemetry captured yet.")
            return OUTPUT_DIR / "telemetry_empty.csv"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = OUTPUT_DIR / f"telemetry_{stamp}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.telemetry[0].keys()))
            writer.writeheader()
            writer.writerows(self.telemetry)
        self.last_export_path = path
        return path

    def generate_performance_chart(self) -> Optional[Path]:
        if not self.telemetry:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        t = [float(r["t"]) for r in self.telemetry]
        towers = [tower.name.lower() for tower in self.towers]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = OUTPUT_DIR / f"performance_{stamp}.png"

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for key in towers:
            axes[0].plot(t, [float(r[f"{key}_rss_dbm"]) for r in self.telemetry], label=f"{key.upper()} RSS")
        axes[0].set_ylabel("RSS (dBm)")
        axes[0].set_title("Tower RSS Over Time")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        tower_indices = {tower.name: idx for idx, tower in enumerate(self.towers)}
        current_series = [tower_indices[str(r["current_tower"])] for r in self.telemetry]
        axes[1].plot(t, current_series, linewidth=1.4, color="#f8c76b", label="Connected tower index")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Tower ID")
        axes[1].set_title("Handoff Decisions Timeline")
        axes[1].set_yticks(list(tower_indices.values()), labels=list(tower_indices.keys()))
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        fig.tight_layout()
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        return chart_path

    def trajectory_for_evaluation(self) -> List[Tuple[int, int]]:
        if self.replay_positions:
            return list(self.replay_positions)
        if self.telemetry:
            return [(int(float(r["ue_x"])), int(float(r["ue_y"]))) for r in self.telemetry]
        return []

    def evaluate_model(self, model: str, positions: List[Tuple[int, int]]) -> Dict[str, object]:
        dt = 1.0 / FPS
        hold_frames = max(1, int(HANDOFF_HOLD_S * FPS))
        kalman = Kalman2D()

        current_tower = self.towers[0]
        candidate_tower: Optional[Tower] = None
        candidate_frames = 0
        handoff_count = 0
        ping_pong = 0
        outage_time = 0.0
        total_rss = 0.0
        history: List[str] = [current_tower.name]
        series_t = []
        series_conn = []
        series_rss = []

        prev_pos = positions[0]
        for idx, pos in enumerate(positions):
            vx = (pos[0] - prev_pos[0]) * METER_PER_PIXEL / dt
            vy = (pos[1] - prev_pos[1]) * METER_PER_PIXEL / dt
            vel = (vx, vy)
            prev_pos = pos
            kalman.update(float(pos[0]), float(pos[1]), dt)

            measurements = {t: self.calculate_measurement(t, pos, vel) for t in self.towers}
            strongest = max(self.towers, key=lambda t: measurements[t].rss_dbm)
            current_rss = measurements[current_tower].rss_dbm
            strongest_rss = measurements[strongest].rss_dbm

            if model == "kalman":
                px, py = kalman.predict(PREDICT_LOOKAHEAD_S)
            else:
                px = pos[0] + (vx / METER_PER_PIXEL) * PREDICT_LOOKAHEAD_S
                py = pos[1] + (vy / METER_PER_PIXEL) * PREDICT_LOOKAHEAD_S
            ppos = self.clamp_ue(int(px), int(py))
            predicted_scores = {
                t: self.calculate_measurement(t, ppos, vel).rss_dbm for t in self.towers
            }
            predicted_best = max(self.towers, key=lambda t: predicted_scores[t])
            predictive_pressure = predicted_scores[predicted_best] - predicted_scores[current_tower] >= HANDOFF_MARGIN_DB

            trigger_by_margin = strongest != current_tower and (strongest_rss - current_rss >= HANDOFF_MARGIN_DB)
            trigger_predictive = predicted_best != current_tower and predictive_pressure
            should_consider = trigger_by_margin or trigger_predictive
            chosen_candidate = strongest if trigger_by_margin else predicted_best

            if should_consider:
                if candidate_tower != chosen_candidate:
                    candidate_tower = chosen_candidate
                    candidate_frames = 1
                else:
                    candidate_frames += 1
                    if candidate_frames >= hold_frames:
                        old = current_tower.name
                        current_tower = chosen_candidate
                        handoff_count += 1
                        history.append(current_tower.name)
                        if len(history) >= 3 and history[-3] == history[-1] and history[-2] != history[-1]:
                            ping_pong += 1
                        candidate_tower = None
                        candidate_frames = 0
                        _ = old
            else:
                candidate_tower = None
                candidate_frames = 0

            conn_rss = measurements[current_tower].rss_dbm
            total_rss += conn_rss
            if conn_rss < OUTAGE_RSS_THRESHOLD_DBM:
                outage_time += dt
            series_t.append(idx * dt)
            series_conn.append(next(i for i, t in enumerate(self.towers) if t.name == current_tower.name))
            series_rss.append(conn_rss)

        n = max(1, len(positions))
        return {
            "model": model,
            "samples": n,
            "handoff_count": handoff_count,
            "ping_pong_count": ping_pong,
            "avg_connected_rss_dbm": round(total_rss / n, 3),
            "outage_time_s": round(outage_time, 3),
            "series_t": series_t,
            "series_conn": series_conn,
            "series_rss": series_rss,
        }

    def run_ab_evaluation(self) -> Optional[str]:
        positions = self.trajectory_for_evaluation()
        if len(positions) < 10:
            return None
        eval_velocity = self.evaluate_model("velocity", positions)
        eval_kalman = self.evaluate_model("kalman", positions)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = OUTPUT_DIR / f"ab_eval_{stamp}.csv"
        fields = [
            "model",
            "samples",
            "handoff_count",
            "ping_pong_count",
            "avg_connected_rss_dbm",
            "outage_time_s",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow({k: eval_velocity[k] for k in fields})
            writer.writerow({k: eval_kalman[k] for k in fields})

        self.generate_ab_chart(eval_velocity, eval_kalman, stamp)
        return csv_path.name

    def generate_ab_chart(self, a: Dict[str, object], b: Dict[str, object], stamp: str) -> Optional[Path]:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        chart_path = OUTPUT_DIR / f"ab_eval_{stamp}.png"
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(a["series_t"], a["series_rss"], label="velocity connected RSS", linewidth=1.3)
        axes[0].plot(b["series_t"], b["series_rss"], label="kalman connected RSS", linewidth=1.3)
        axes[0].set_ylabel("Connected RSS (dBm)")
        axes[0].set_title("A/B Predictor Comparison")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(a["series_t"], a["series_conn"], label="velocity tower index", linewidth=1.2)
        axes[1].plot(b["series_t"], b["series_conn"], label="kalman tower index", linewidth=1.2)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Connected tower index")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        fig.tight_layout()
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        return chart_path

    def draw_grid(self) -> None:
        self.screen.fill(CITY_BG)
        for x in range(0, WIDTH, 40):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, HEIGHT), 1)
        for y in range(0, HEIGHT, 40):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WIDTH, y), 1)

    def draw_buildings(self) -> None:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for b in self.buildings:
            r = b.to_rect()
            pygame.draw.rect(overlay, (*BUILDING_COLOR, BUILDING_ALPHA), r, border_radius=4)
            pygame.draw.rect(overlay, BUILDING_BORDER, r, 2, border_radius=4)
        self.screen.blit(overlay, (0, 0))

        if self.drag_building_start:
            sx, sy = self.drag_building_start
            ex, ey = pygame.mouse.get_pos()
            temp = Building(float(sx), float(sy), float(ex - sx), float(ey - sy)).to_rect()
            pygame.draw.rect(self.screen, (210, 220, 230), temp, 2, border_radius=4)

        for blocker in self.dynamic_blockers:
            if blocker.kind == "pedestrian":
                color = (235, 180, 72)
            elif blocker.kind == "bus":
                color = (240, 125, 120)
            else:
                color = (176, 146, 255)
            center = (int(blocker.x), int(blocker.y))
            pygame.draw.circle(self.screen, color, center, int(blocker.radius))
            pygame.draw.circle(self.screen, (245, 245, 252), center, int(blocker.radius), 2)
            label = blocker.kind[0].upper()
            self.screen.blit(self.small_font.render(label, True, (248, 248, 255)), (center[0] - 4, center[1] - 8))
        for relay in self.relays:
            center = (int(relay.x), int(relay.y))
            pygame.draw.polygon(
                self.screen,
                (124, 238, 246),
                [(center[0], center[1] - 10), (center[0] + 8, center[1] + 8), (center[0] - 8, center[1] + 8)],
            )
            self.screen.blit(self.small_font.render("R", True, (210, 252, 255)), (center[0] - 4, center[1] - 26))

    def draw_links_and_towers(self, measurements: Dict[Tower, Measurement]) -> None:
        for tower in self.towers:
            m = measurements[tower]
            line_color = GOOD_COLOR if not m.blocked else BAD_COLOR
            pygame.draw.line(self.screen, line_color, (int(tower.x), int(tower.y)), self.ue_pos, 2)
            radius = 12 if tower != self.current_tower else 16
            pygame.draw.circle(self.screen, tower.color, (int(tower.x), int(tower.y)), radius)
            pygame.draw.circle(self.screen, (240, 240, 255), (int(tower.x), int(tower.y)), radius, 2)
            self.screen.blit(self.small_font.render(tower.name, True, tower.color), (int(tower.x) + 12, int(tower.y) - 26))
        if self.connection_mode == "relay":
            relay = next((r for r in self.relays if r.name == self.relay_name), None)
            relay_tower = next((t for t in self.towers if t.name == self.relay_tower_name), None)
            if relay and relay_tower:
                pygame.draw.line(self.screen, (96, 255, 244), self.ue_pos, (int(relay.x), int(relay.y)), 3)
                pygame.draw.line(self.screen, (96, 255, 244), (int(relay.x), int(relay.y)), (int(relay_tower.x), int(relay_tower.y)), 3)

    def draw_ue(self) -> None:
        pygame.draw.circle(self.screen, UE_RING, self.ue_pos, 14, 2)
        pygame.draw.circle(self.screen, UE_COLOR, self.ue_pos, 7)
        self.screen.blit(self.small_font.render("UE", True, TEXT_COLOR), (self.ue_pos[0] + 10, self.ue_pos[1] + 10))

    def draw_dashboard(self, measurements: Dict[Tower, Measurement]) -> None:
        panel = pygame.Rect(WIDTH - 430, 16, 414, HEIGHT - 32)
        pygame.draw.rect(self.screen, PANEL_BG, panel, border_radius=12)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel, 2, border_radius=12)

        self.screen.blit(self.title_font.render("V2X Smart Handoff Enterprise", True, TEXT_COLOR), (panel.x + 12, panel.y + 12))
        speed = math.hypot(self.ue_velocity[0], self.ue_velocity[1])
        self.screen.blit(self.small_font.render(f"Connected: {self.current_tower.name}", True, GOOD_COLOR), (panel.x + 16, panel.y + 52))
        self.screen.blit(self.small_font.render(f"UE speed: {speed:6.2f} m/s", True, TEXT_COLOR), (panel.x + 16, panel.y + 74))
        self.screen.blit(self.small_font.render(f"Predictive model: {self.predictive_mode}", True, WARN_COLOR), (panel.x + 16, panel.y + 96))
        self.screen.blit(self.small_font.render(f"Replay: {'ON' if self.replay_mode else 'OFF'}", True, WARN_COLOR), (panel.x + 16, panel.y + 118))
        self.screen.blit(
            self.small_font.render(f"Packet={self.packet_profile} | Weather={self.weather_mode}", True, (158, 205, 255)),
            (panel.x + 16, panel.y + 140),
        )
        self.screen.blit(
            self.small_font.render(f"AoI={self.aoi_age_s:0.3f}s / {self.deadline_for_profile():0.3f}s", True, (186, 219, 255)),
            (panel.x + 16, panel.y + 160),
        )
        mode_text = f"Mode: {self.connection_mode.upper()}"
        if self.connection_mode == "relay":
            mode_text += f" ({self.relay_name}->{self.relay_tower_name})"
        self.screen.blit(self.small_font.render(mode_text, True, (143, 248, 226)), (panel.x + 16, panel.y + 180))

        y = panel.y + 204
        for tower in self.towers:
            m = measurements[tower]
            color = tower.color if tower != self.current_tower else GOOD_COLOR
            text = f"{tower.name} | d={m.distance_m:6.1f}m | RSS={m.rss_dbm:6.1f} dBm | fd={m.doppler_hz:7.1f}Hz"
            self.screen.blit(self.small_font.render(text, True, color), (panel.x + 12, y))
            y += 22
            risk = self.last_risk_scores.get(tower.name, 0.0)
            load = self.last_load_scores.get(tower.name, 0.0)
            aoi = self.last_aoi_scores.get(tower.name, 0.0)
            self.screen.blit(
                self.small_font.render(f"  surv={risk:0.3f} load={load:0.2f} aoi={aoi:0.2f}", True, (172, 193, 236)),
                (panel.x + 12, y),
            )
            y += 18
            if m.blocked:
                self.screen.blit(self.small_font.render("  LOS blocked by building/blocker", True, BAD_COLOR), (panel.x + 12, y))
                y += 18

        y += 8
        self.screen.blit(self.font.render("Event Log", True, TEXT_COLOR), (panel.x + 16, y))
        y += 26
        for entry in self.events:
            self.screen.blit(self.small_font.render(entry, True, WARN_COLOR), (panel.x + 16, y))
            y += 18

    def draw_help(self) -> None:
        if not self.show_help:
            return
        lines = self.get_help_lines()
        h = 24 + len(lines) * 19
        help_rect = pygame.Rect(14, 14, 470, h)
        pygame.draw.rect(self.screen, (10, 15, 23), help_rect, border_radius=10)
        pygame.draw.rect(self.screen, (82, 104, 128), help_rect, 2, border_radius=10)
        y = help_rect.y + 8
        for idx, line in enumerate(lines):
            font = self.font if idx == 0 else self.small_font
            color = TEXT_COLOR if idx == 0 else (189, 205, 225)
            self.screen.blit(font.render(line, True, color), (help_rect.x + 10, y))
            y += 19

    def draw(self, measurements: Dict[Tower, Measurement]) -> None:
        self.draw_grid()
        self.draw_buildings()
        self.draw_links_and_towers(measurements)
        self.draw_ue()
        self.draw_dashboard(measurements)
        self.draw_help()
        pygame.display.flip()

    def run(self) -> None:
        self.push_event("Simulator started.")
        while self.running:
            dt = (self.clock.tick(FPS) / 1000.0) if not self.headless else (1.0 / FPS)
            if not self.headless:
                self.handle_events()
            self.update_ue_kinematics(dt)
            measurements = self.update_handoff_logic()
            self.update_aoi_state(measurements, dt)
            self.record_telemetry(measurements)
            if not self.headless:
                self.draw(measurements)

        if self.telemetry and not self.last_export_path:
            path = self.export_telemetry()
            self.push_event(f"Auto-exported telemetry: {path.name}")
        pygame.quit()


def main() -> None:
    if not run_start_screen():
        return
    simulator = V2XSmartHandoffSimulator()
    simulator.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)
