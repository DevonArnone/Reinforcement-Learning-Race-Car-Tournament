"""Tracks are stored as a handful of control points; everything else is derived.

The control points are smoothed with a closed Catmull-Rom spline and resampled
at an even spacing. From that centerline we get the road edges, the distance
along the lap for any position, and the start grid. No images are involved, so
a new track is just a new JSON file.
"""
import json
from dataclasses import dataclass

import numpy as np

import config


def catmull_rom_loop(points: np.ndarray, steps: int = 24) -> np.ndarray:
    n = len(points)
    t = np.linspace(0, 1, steps, endpoint=False)[:, None]
    pieces = []
    for i in range(n):
        p0, p1, p2, p3 = points[i - 1], points[i], points[(i + 1) % n], points[(i + 2) % n]
        pieces.append(0.5 * (
            2 * p1
            + (p2 - p0) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
            + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3
        ))
    return np.vstack(pieces)


def resample_loop(points: np.ndarray, spacing: float) -> np.ndarray:
    closed = np.vstack([points, points[:1]])
    cumulative = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    count = int(cumulative[-1] // spacing)
    s = np.linspace(0, cumulative[-1], count, endpoint=False)
    return np.column_stack([np.interp(s, cumulative, closed[:, 0]),
                            np.interp(s, cumulative, closed[:, 1])])


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class GridSlot:
    position: np.ndarray
    heading: float
    distance_behind: float


class Track:
    def __init__(self, name: str, width: float, control_points):
        self.name = name
        self.width = float(width)
        self.control_points = np.asarray(control_points, dtype=float)

        self.center = resample_loop(catmull_rom_loop(self.control_points), config.SAMPLE_SPACING)
        self.count = len(self.center)

        segments = np.roll(self.center, -1, axis=0) - self.center
        self.segment_length = np.linalg.norm(segments, axis=1)
        self.distance = np.concatenate([[0], np.cumsum(self.segment_length)[:-1]])
        self.length = float(self.segment_length.sum())

        # Central difference gives a smoother tangent than the forward segment.
        tangent = np.roll(self.center, -1, axis=0) - np.roll(self.center, 1, axis=0)
        self.tangent = tangent / np.linalg.norm(tangent, axis=1, keepdims=True)
        self.heading = np.arctan2(self.tangent[:, 1], self.tangent[:, 0])
        normal = np.column_stack([-self.tangent[:, 1], self.tangent[:, 0]])
        self.left = self.center + normal * self.width / 2
        self.right = self.center - normal * self.width / 2

    @classmethod
    def load(cls, name: str) -> "Track":
        data = json.loads((config.TRACKS_DIR / f"{name}.json").read_text())
        return cls(data["name"], data["width"], data["points"])

    @staticmethod
    def available() -> list[str]:
        return sorted(p.stem for p in config.TRACKS_DIR.glob("*.json"))

    def bounds(self, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        edges = np.vstack([self.left, self.right])
        return edges.min(axis=0) - margin, edges.max(axis=0) + margin

    def locate(self, positions: np.ndarray, hints: np.ndarray):
        """Project positions onto the centerline, searching only near each hint index.

        Returns (segment index, distance along the lap, distance from centerline).
        Searching a small window keeps this cheap and stops a car from snapping
        to a different part of the track that happens to pass close by.
        """
        window = (hints[:, None] + np.arange(-config.SEARCH_WINDOW, config.SEARCH_WINDOW + 1)) % self.count
        a = self.center[window]
        ab = self.center[(window + 1) % self.count] - a
        ap = positions[:, None, :] - a
        t = np.clip((ap * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)
        gap = np.linalg.norm(ap - ab * t[..., None], axis=-1)

        best = gap.argmin(axis=1)
        rows = np.arange(len(positions))
        segment = window[rows, best]
        along = (self.distance[segment] + t[rows, best] * self.segment_length[segment]) % self.length
        return segment, along, gap[rows, best]

    def nearest_index(self, position) -> int:
        return int(np.linalg.norm(self.center - position, axis=1).argmin())

    def index_at(self, along):
        return (np.searchsorted(self.distance, np.mod(along, self.length), side="right") - 1) % self.count

    def heading_at(self, along):
        return self.heading[self.index_at(along)]

    def start_grid(self, cars: int, row_gap: float = 42.0) -> list[GridSlot]:
        """Two-wide staggered grid behind the start line."""
        slots = []
        for i in range(cars):
            behind = 30 + i * row_gap / 2
            along = self.length - behind
            index = self.index_at(along)
            side = 1 if i % 2 == 0 else -1
            normal = np.array([-self.tangent[index, 1], self.tangent[index, 0]])
            position = self.center[index] + normal * side * self.width / 4
            slots.append(GridSlot(position, float(self.heading[index]), behind))
        return slots
