"""All cars on a track, stored as arrays so a whole generation updates at once."""
import numpy as np

import config
from track import Track


class Fleet:
    def __init__(self, track: Track, count: int):
        self.track = track
        self.count = count
        self.position = np.zeros((count, 2))
        self.heading = np.zeros(count)
        self.speed = np.zeros(count)
        self.active = np.ones(count, dtype=bool)   # cars that are allowed to move this step
        self.off_track = np.zeros(count, dtype=bool)

        # Lap progress is tracked as unwrapped distance so that driving
        # backwards over the line can never count as a lap.
        self.segment = np.zeros(count, dtype=int)
        self.along = np.zeros(count)
        self.progress = np.zeros(count)

    def place(self, i: int, position, heading: float, progress: float = 0.0):
        self.position[i] = position
        self.heading[i] = heading
        self.speed[i] = 0.0
        self.segment[i] = self.track.nearest_index(position)
        _, along, _ = self.track.locate(self.position[i:i + 1], self.segment[i:i + 1])
        self.along[i] = along[0]
        self.progress[i] = progress
        self.off_track[i] = False

    def place_all_at(self, along: float = 0.0):
        index = self.track.index_at(along)
        for i in range(self.count):
            self.place(i, self.track.center[index], self.track.heading[index])

    def drive(self, steer: np.ndarray, throttle: np.ndarray, dt: float = config.DT):
        """Arcade physics: speed along the heading, turn rate limited by speed."""
        steer = np.clip(steer, -1, 1)
        throttle = np.clip(throttle, -1, 1)
        moving = self.active

        accel = np.where(throttle >= 0, throttle * config.ACCELERATION, throttle * config.BRAKING)
        speed = self.speed + (accel - self.speed * config.ROLLING_DRAG) * dt
        speed = np.clip(speed, 0.0, config.MAX_SPEED)

        # Cars can't turn on the spot, and they lose a bit of grip when fast.
        speed_ratio = speed / config.MAX_SPEED
        steering_authority = np.clip(speed / config.FULL_STEER_SPEED, 0, 1) * (1 - config.HIGH_SPEED_GRIP_LOSS * speed_ratio)
        heading = self.heading + steer * config.MAX_TURN_RATE * steering_authority * dt

        direction = np.column_stack([np.cos(heading), np.sin(heading)])
        position = self.position + direction * speed[:, None] * dt

        self.speed = np.where(moving, speed, self.speed)
        self.heading = np.where(moving, heading, self.heading)
        self.position = np.where(moving[:, None], position, self.position)
        self._update_progress(moving)

    def _update_progress(self, moving: np.ndarray):
        segment, along, gap = self.track.locate(self.position, self.segment)
        delta = along - self.along
        half_lap = self.track.length / 2
        delta = np.where(delta < -half_lap, delta + self.track.length, delta)
        delta = np.where(delta > half_lap, delta - self.track.length, delta)

        self.progress = np.where(moving, self.progress + delta, self.progress)
        self.segment = np.where(moving, segment, self.segment)
        self.along = np.where(moving, along, self.along)
        self.off_track = moving & (gap > self.track.width / 2 - config.CAR_WIDTH / 2)

    @property
    def laps(self) -> np.ndarray:
        return np.floor(self.progress / self.track.length).astype(int)
