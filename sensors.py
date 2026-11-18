"""What a car can "see": distances to the walls along a fan of rays, plus a few
readings about its own speed and the road ahead."""
import numpy as np

import config
from car import Fleet
from track import wrap_angle

RAY_COUNT = 9
RAY_LENGTH = 250.0
RAY_SPREAD = np.radians(150)
RAY_ANGLES = np.linspace(-RAY_SPREAD / 2, RAY_SPREAD / 2, RAY_COUNT)
LOOKAHEAD = 150.0

# Rays only test walls near the car. 45 samples either side covers the ray
# length even on the outside of a corner, where the edge is longer than the centerline.
WALL_WINDOW = 45

INPUT_COUNT = RAY_COUNT + 3


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def cast_rays(fleet: Fleet) -> np.ndarray:
    """Distance to the nearest wall along each ray, in pixels. Shape (cars, rays)."""
    track = fleet.track
    window = (fleet.segment[:, None] + np.arange(-WALL_WINDOW, WALL_WINDOW + 1)) % track.count
    following = (window + 1) % track.count
    starts = np.concatenate([track.left[window], track.right[window]], axis=1)
    ends = np.concatenate([track.left[following], track.right[following]], axis=1)

    angles = fleet.heading[:, None] + RAY_ANGLES
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    # Ray: origin + t * direction. Wall: start + u * (end - start). Solve for t and u.
    origin = fleet.position[:, None, None, :]
    direction = directions[:, :, None, :]
    wall = (ends - starts)[:, None, :, :]
    to_start = starts[:, None, :, :] - origin

    denominator = _cross(direction, wall)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross(to_start, wall) / denominator
        u = _cross(to_start, direction) / denominator
    hit = (np.abs(denominator) > 1e-9) & (t >= 0) & (u >= 0) & (u <= 1)
    t = np.where(hit, t, RAY_LENGTH)
    return np.minimum(t.min(axis=2), RAY_LENGTH)


def ray_endpoints(fleet: Fleet, distances: np.ndarray) -> np.ndarray:
    angles = fleet.heading[:, None] + RAY_ANGLES
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    return fleet.position[:, None, :] + directions * distances[..., None]


def observe(fleet: Fleet, distances: np.ndarray) -> np.ndarray:
    """Network inputs, all roughly in [-1, 1]."""
    track = fleet.track
    rays = distances / RAY_LENGTH
    speed = fleet.speed / config.MAX_SPEED
    here = track.heading_at(fleet.along)
    heading_error = wrap_angle(fleet.heading - here) / (np.pi / 2)
    bend_ahead = wrap_angle(track.heading_at(fleet.along + LOOKAHEAD) - here) / (np.pi / 2)
    return np.column_stack([rays, speed, heading_error, bend_ahead])
