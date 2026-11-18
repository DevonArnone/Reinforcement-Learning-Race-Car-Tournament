import numpy as np
import pygame

import config
from car import Fleet
from track import Track, wrap_angle

GRASS = (64, 122, 66)
GRASS_STRIPE = (70, 131, 71)
ASPHALT = (54, 56, 61)
EDGE_LINE = (226, 226, 220)
CURB_RED = (196, 52, 46)
TEXT = (238, 238, 232)
MUTED = (160, 164, 170)
PANEL = (16, 18, 22, 190)

CAR_COLORS = [
    (232, 72, 60), (64, 140, 235), (245, 190, 50), (90, 200, 120),
    (180, 100, 220), (240, 130, 40), (60, 200, 210), (230, 230, 230),
]

TRACK_MARGIN = 80
CURB_RADIUS = 220
_fonts: dict[int, pygame.font.Font] = {}


def font(size: int) -> pygame.font.Font:
    if size not in _fonts:
        _fonts[size] = pygame.font.Font(None, size)
    return _fonts[size]


SCREEN_SIZE = np.array([config.SCREEN_W, config.SCREEN_H], dtype=float)


def fit_zoom(track: Track) -> float:
    """Zoom at which the whole track fits on screen (never above 1)."""
    low, high = track.bounds(30)
    return float(min(1.0, *(SCREEN_SIZE / (high - low))))


def build_track_surface(track: Track) -> tuple[pygame.Surface, np.ndarray]:
    """Draw the track once at full scale. Returns the surface and its world origin."""
    low, high = track.bounds(TRACK_MARGIN)
    # Pad out so the grass still fills the window when zoomed out to fit.
    padding = np.maximum(SCREEN_SIZE / fit_zoom(track) - (high - low), 0) / 2
    low, high = low - padding, high + padding
    size = np.ceil(high - low).astype(int)
    surface = pygame.Surface(size)
    surface.fill(GRASS)
    for y in range(0, size[1], 80):
        surface.fill(GRASS_STRIPE, (0, y, size[0], 40))

    left = track.left - low
    right = track.right - low
    n = track.count
    for i in range(n):
        j = (i + 1) % n
        pygame.draw.polygon(surface, ASPHALT, [left[i], left[j], right[j], right[i]])

    # Curbs only on tight corners; a plain white line everywhere else.
    turn = np.abs(wrap_angle(np.roll(track.heading, -1) - track.heading))
    radius = track.segment_length / np.maximum(turn, 1e-6)
    for edge in (left, right):
        for i in range(n):
            j = (i + 1) % n
            if radius[i] < CURB_RADIUS:
                color = CURB_RED if (i // 2) % 2 else EDGE_LINE
                pygame.draw.line(surface, color, edge[i], edge[j], 7)
            else:
                pygame.draw.line(surface, EDGE_LINE, edge[i], edge[j], 3)

    _draw_start_line(surface, track, low)
    return surface, low


def _draw_start_line(surface: pygame.Surface, track: Track, origin: np.ndarray):
    center = track.center[0] - origin
    forward = track.tangent[0]
    across = np.array([-forward[1], forward[0]])
    square = 9
    cells = int(track.width // square)
    for row in range(2):
        for k in range(cells):
            color = (245, 245, 245) if (row + k) % 2 else (20, 20, 20)
            offset = across * (k - cells / 2) * square + forward * row * square
            corner = center + offset
            pts = [corner, corner + across * square, corner + across * square + forward * square, corner + forward * square]
            pygame.draw.polygon(surface, color, pts)


def make_car_sprite(color) -> pygame.Surface:
    sprite = pygame.Surface((config.CAR_LENGTH, config.CAR_WIDTH), pygame.SRCALPHA)
    body = sprite.get_rect()
    pygame.draw.rect(sprite, color, body, border_radius=4)
    pygame.draw.rect(sprite, (20, 22, 26), (body.w * 0.45, 2, body.w * 0.25, body.h - 4), border_radius=2)
    pygame.draw.rect(sprite, (250, 250, 250), (body.w - 4, 2, 3, body.h - 4))
    return sprite


class Camera:
    """World-to-screen transform. Small tracks are framed whole; big ones are followed."""

    def __init__(self, track: Track, surface: pygame.Surface, origin: np.ndarray):
        self.view = SCREEN_SIZE
        self.origin = origin
        self.world_size = np.array(surface.get_size(), dtype=float)
        self.fit_zoom = fit_zoom(track)
        self.follow_zoom = 1.0
        self.overview = self.fit_zoom > 0.8
        low, high = track.bounds(30)
        self.track_center = (low + high) / 2
        self.center = self.track_center
        self._base = surface
        self._scaled: dict[float, pygame.Surface] = {}

    @property
    def zoom(self) -> float:
        return self.fit_zoom if self.overview else self.follow_zoom

    def toggle_overview(self):
        if self.fit_zoom <= 0.8:
            self.overview = not self.overview

    def follow(self, target, smoothing: float = 0.12):
        if self.overview:
            self.center = self.track_center
            return
        self.center = self.center + (np.asarray(target) - self.center) * smoothing
        half_view = self.view / (2 * self.zoom)
        low = self.origin + half_view
        high = self.origin + self.world_size - half_view
        self.center = np.clip(self.center, low, np.maximum(low, high))

    def to_screen(self, world):
        return (np.asarray(world) - self.center) * self.zoom + self.view / 2

    def draw_track(self, screen: pygame.Surface):
        zoom = round(self.zoom, 3)
        if zoom not in self._scaled:
            self._scaled[zoom] = self._base if zoom == 1 else pygame.transform.smoothscale_by(self._base, zoom)
        screen.blit(self._scaled[zoom], self.to_screen(self.origin))


def draw_cars(screen, camera: Camera, fleet: Fleet, sprites: list[pygame.Surface], faded=None, on_top=None):
    """Draw each car with sprites[i % len(sprites)]. `on_top` is drawn last so it stays visible."""
    order = [i for i in range(fleet.count) if i != on_top] + ([on_top] if on_top is not None else [])
    for i in order:
        sprite = pygame.transform.rotozoom(sprites[i % len(sprites)], -np.degrees(fleet.heading[i]), camera.zoom)
        if faded is not None and faded[i]:
            sprite.set_alpha(70)
        screen.blit(sprite, sprite.get_rect(center=camera.to_screen(fleet.position[i])))


def draw_panel(screen, lines: list[str], topleft=(16, 16), size: int = 24, width: int | None = None):
    rendered = [font(size).render(line, True, TEXT) for line in lines]
    w = width or max(r.get_width() for r in rendered) + 24
    h = sum(r.get_height() + 4 for r in rendered) + 16
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill(PANEL)
    screen.blit(panel, topleft)
    y = topleft[1] + 10
    for r in rendered:
        screen.blit(r, (topleft[0] + 12, y))
        y += r.get_height() + 4


def format_time(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}:{rest:06.3f}" if minutes else f"{rest:.3f}s"


def draw_hint(screen, text: str):
    label = font(20).render(text, True, MUTED)
    screen.blit(label, label.get_rect(midbottom=(config.SCREEN_W / 2, config.SCREEN_H - 10)))


def draw_fitness_graph(screen, history, size=(260, 110)):
    """Best and mean fitness over the most recent generations."""
    if len(history) < 2:
        return
    recent = history[-60:]
    w, h = size
    x0, y0 = 16, config.SCREEN_H - h - 40
    panel = pygame.Surface(size, pygame.SRCALPHA)
    panel.fill(PANEL)
    screen.blit(panel, (x0, y0))

    best = np.array([r.best_fitness for r in recent])
    mean = np.array([r.mean_fitness for r in recent])
    low, high = min(mean.min(), 0), max(best.max(), 1)
    xs = x0 + 10 + np.linspace(0, w - 20, len(recent))

    def to_y(values):
        return y0 + h - 22 - (values - low) / (high - low) * (h - 34)

    pygame.draw.lines(screen, MUTED, False, list(zip(xs, to_y(mean))), 2)
    pygame.draw.lines(screen, (250, 220, 80), False, list(zip(xs, to_y(best))), 2)
    screen.blit(font(18).render("fitness  (best / mean)", True, MUTED), (x0 + 10, y0 + h - 18))
