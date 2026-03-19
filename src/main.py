import csv
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pygame


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

HANDOFF_MARGIN_DB = 5.0
HANDOFF_HOLD_S = 0.5
PREDICT_LOOKAHEAD_S = 0.8
PREDICT_DT_S = 0.1

OUTPUT_DIR = Path("outputs")


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


class V2XSmartHandoffSimulator:
    def __init__(self) -> None:
        pygame.init()
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
        self.drag_building_start: Optional[Tuple[int, int]] = None

        self.ue_pos = (WIDTH // 2, HEIGHT // 2)
        self.prev_ue_pos = self.ue_pos
        self.ue_velocity = (0.0, 0.0)

        self.current_tower: Tower = self.towers[0]
        self.candidate_tower: Optional[Tower] = None
        self.candidate_since = 0.0

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

    def apply_scenario(self, name: str) -> None:
        if name == "baseline":
            self.buildings = []
            self.ue_pos = (WIDTH // 2, HEIGHT // 2)
            self.prev_ue_pos = self.ue_pos
        elif name == "urban_canyon":
            self.buildings = [
                Building(460, 80, 110, 260),
                Building(640, 180, 130, 310),
                Building(500, 560, 260, 120),
            ]
            self.ue_pos = (220, 520)
            self.prev_ue_pos = self.ue_pos
        elif name == "intersection":
            self.buildings = [
                Building(420, 280, 170, 120),
                Building(690, 280, 170, 120),
                Building(550, 450, 160, 120),
            ]
            self.ue_pos = (640, 680)
            self.prev_ue_pos = self.ue_pos
        self.push_event(f"Scenario loaded: {name}")

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

    def calculate_measurement(self, tower: Tower, ue_pos: Tuple[float, float], ue_vel: Tuple[float, float]) -> Measurement:
        dx = ue_pos[0] - tower.x
        dy = ue_pos[1] - tower.y
        distance_px = max(1.0, math.hypot(dx, dy))
        distance_m = max(0.5, distance_px * METER_PER_PIXEL)
        path_loss_db = REFERENCE_LOSS_DB + 10.0 * PATH_LOSS_EXPONENT * math.log10(distance_m / REFERENCE_DISTANCE_M)
        rss_dbm = TX_POWER_DBM - path_loss_db

        blocked = any(self.line_intersects_rect((tower.x, tower.y), ue_pos, b) for b in self.buildings)
        if blocked:
            rss_dbm -= SHADOWING_DROP_DB

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
        m = self.calculate_measurement(tower, (px, py), self.ue_velocity)
        return m.rss_dbm

    def update_ue_kinematics(self, dt: float) -> None:
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
        predicted_best_tower = max(self.towers, key=lambda t: predicted_scores[t])
        predictive_pressure = predicted_scores[predicted_best_tower] - predicted_current >= HANDOFF_MARGIN_DB

        trigger_by_margin = strongest != self.current_tower and (strongest_rss - current_rss >= HANDOFF_MARGIN_DB)
        trigger_predictive = predicted_best_tower != self.current_tower and predictive_pressure

        chosen_candidate = strongest if trigger_by_margin else predicted_best_tower
        should_consider = trigger_by_margin or trigger_predictive

        if should_consider:
            if self.candidate_tower != chosen_candidate:
                self.candidate_tower = chosen_candidate
                self.candidate_since = time.time()
            else:
                if time.time() - self.candidate_since >= HANDOFF_HOLD_S:
                    old_name = self.current_tower.name
                    self.current_tower = chosen_candidate
                    self.candidate_tower = None
                    tag = "PREDICTIVE " if trigger_predictive and not trigger_by_margin else ""
                    self.push_event(f"{tag}HANDOFF {old_name} -> {self.current_tower.name}")
        else:
            self.candidate_tower = None
        return measurements

    def record_telemetry(self, measurements: Dict[Tower, Measurement]) -> None:
        elapsed = time.time() - self.start_ts
        row: Dict[str, object] = {
            "t": round(elapsed, 4),
            "ue_x": self.ue_pos[0],
            "ue_y": self.ue_pos[1],
            "ue_vx_mps": round(self.ue_velocity[0], 4),
            "ue_vy_mps": round(self.ue_velocity[1], 4),
            "current_tower": self.current_tower.name,
            "candidate_tower": self.candidate_tower.name if self.candidate_tower else "",
            "predictive_model": self.predictive_mode,
            "replay_mode": int(self.replay_mode),
        }
        for tower in self.towers:
            m = measurements[tower]
            key = tower.name.lower()
            row[f"{key}_rss_dbm"] = round(m.rss_dbm, 4)
            row[f"{key}_doppler_hz"] = round(m.doppler_hz, 4)
            row[f"{key}_distance_m"] = round(m.distance_m, 4)
            row[f"{key}_blocked"] = int(m.blocked)
        self.telemetry.append(row)

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

    def draw_links_and_towers(self, measurements: Dict[Tower, Measurement]) -> None:
        for tower in self.towers:
            m = measurements[tower]
            line_color = GOOD_COLOR if not m.blocked else BAD_COLOR
            pygame.draw.line(self.screen, line_color, (int(tower.x), int(tower.y)), self.ue_pos, 2)
            radius = 12 if tower != self.current_tower else 16
            pygame.draw.circle(self.screen, tower.color, (int(tower.x), int(tower.y)), radius)
            pygame.draw.circle(self.screen, (240, 240, 255), (int(tower.x), int(tower.y)), radius, 2)
            self.screen.blit(self.small_font.render(tower.name, True, tower.color), (int(tower.x) + 12, int(tower.y) - 26))

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

        y = panel.y + 150
        for tower in self.towers:
            m = measurements[tower]
            color = tower.color if tower != self.current_tower else GOOD_COLOR
            text = f"{tower.name} | d={m.distance_m:6.1f}m | RSS={m.rss_dbm:6.1f} dBm | fd={m.doppler_hz:7.1f}Hz"
            self.screen.blit(self.small_font.render(text, True, color), (panel.x + 12, y))
            y += 22
            if m.blocked:
                self.screen.blit(self.small_font.render("  LOS blocked by building", True, BAD_COLOR), (panel.x + 12, y))
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
        lines = [
            "Controls:",
            "- Move mouse: drive UE",
            "- Drag LMB: draw building",
            "- 1/2/3: load scenario preset",
            "- K: switch predictor (velocity/kalman)",
            "- X: export telemetry CSV",
            "- P: generate performance chart",
            "- R: toggle replay from latest CSV",
            "- C: clear buildings | H: toggle help | ESC: quit",
        ]
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
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            self.update_ue_kinematics(dt)
            measurements = self.update_handoff_logic()
            self.record_telemetry(measurements)
            self.draw(measurements)

        if self.telemetry and not self.last_export_path:
            path = self.export_telemetry()
            self.push_event(f"Auto-exported telemetry: {path.name}")
        pygame.quit()


def main() -> None:
    simulator = V2XSmartHandoffSimulator()
    simulator.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)
