"""AI-only races between saved drivers from different points in training."""
from dataclasses import dataclass

import numpy as np

import brain
import config
import sensors
from car import Fleet
from evolution import CHAMPION_PATH
from track import Track

DRIVER_NAMES = ["Vega", "Okafor", "Lindqvist", "Tanaka", "Moreau", "Silva", "Kowalski", "Reyes"]


@dataclass
class Driver:
    name: str
    color: tuple[int, int, int]
    genome: np.ndarray
    generation: int
    pace: float = 1.0  # scales throttle, so copies of one brain still race differently


def drivers_dir():
    """Your own trained drivers if there are any, otherwise the ones in the repo."""
    if CHAMPION_PATH.exists():
        return CHAMPION_PATH.parent
    if (config.PRETRAINED_DIR / "champion.npz").exists():
        return config.PRETRAINED_DIR
    return None


def load_drivers(colors, count: int = 6) -> list[Driver]:
    """The champion plus snapshots spread across its training history.

    Older snapshots are slower and make more mistakes, which gives a natural
    spread of skill without hand-tuning any opponents.
    """
    folder = drivers_dir()
    if folder is None:
        return []
    entries = [brain.load(folder / "champion.npz")]
    snapshots = sorted((folder / "snapshots").glob("gen_*.npz"))
    champion_generation = entries[0][1]["generation"]
    snapshots = [p for p in snapshots if int(p.stem[4:]) < champion_generation]
    if snapshots:
        picks = np.unique(np.linspace(0, len(snapshots) - 1, min(count - 1, len(snapshots))).round().astype(int))
        entries += [brain.load(snapshots[i]) for i in picks[::-1]]

    drivers = []
    for i in range(count):
        genome, info = entries[i % len(entries)]
        repeat = i // len(entries)
        drivers.append(Driver(DRIVER_NAMES[i], colors[i], genome, info["generation"], pace=1.0 - 0.04 * repeat))
    return drivers


class Race:
    COUNTDOWN = 3.0
    CRASH_PENALTY = 1.0   # seconds parked after a crash
    STALL_SECONDS = 3.0
    FINISH_GRACE = 15.0   # how long the rest of the field gets after the winner
    CONTACT_DISTANCE = config.CAR_LENGTH * 0.75

    def __init__(self, track: Track, drivers: list[Driver], laps: int = 3, seed: int | None = None):
        self.track = track
        self.drivers = drivers
        self.laps = laps
        n = len(drivers)
        self.genomes = np.array([d.genome for d in drivers])
        self.pace = np.array([d.pace for d in drivers])

        # Trained drivers are close in pace, so the grid decides a lot. A shuffled
        # grid means every restart is a different race.
        self.grid = np.random.default_rng(seed).permutation(n)
        self.fleet = Fleet(track, n)
        for slot, i in zip(track.start_grid(n), self.grid):
            self.fleet.place(i, slot.position, slot.heading, -slot.distance_behind)

        self.countdown = self.COUNTDOWN
        self.clock = 0.0
        self.rays = np.zeros((n, sensors.RAY_COUNT))
        self.completed = np.zeros(n, dtype=int)
        self.lap_started = np.zeros(n)
        self.last_lap = np.full(n, np.nan)
        self.best_lap = np.full(n, np.nan)
        self.finish_time = np.full(n, np.nan)
        self.parked_until = np.zeros(n)
        self.slow_since = np.zeros(n)
        self.crashes = np.zeros(n, dtype=int)
        self.pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    @property
    def started(self) -> bool:
        return self.countdown <= 0

    @property
    def finished(self) -> np.ndarray:
        return ~np.isnan(self.finish_time)

    @property
    def over(self) -> bool:
        if self.finished.all():
            return True
        return self.finished.any() and self.clock > np.nanmin(self.finish_time) + self.FINISH_GRACE

    def step(self, dt: float = config.DT):
        if not self.started:
            self.countdown -= dt
            self.rays = sensors.cast_rays(self.fleet)
            return
        fleet = self.fleet
        self.clock += dt
        fleet.active = ~self.finished & (self.clock >= self.parked_until)

        self.rays = sensors.cast_rays(fleet)
        outputs = brain.forward(self.genomes, sensors.observe(fleet, self.rays))
        throttle = np.where(outputs[:, 1] > 0, outputs[:, 1] * self.pace, outputs[:, 1])
        fleet.drive(outputs[:, 0], throttle, dt)
        self._resolve_contact()

        self.slow_since = np.where(fleet.speed > 20, self.clock, self.slow_since)
        stuck = fleet.active & (self.clock - self.slow_since > self.STALL_SECONDS)
        for i in np.flatnonzero(fleet.off_track | stuck):
            self._recover(i)
        self._count_laps()

    def _resolve_contact(self):
        """Cars that overlap get pushed apart, and the one behind loses some speed.

        Without this the whole field drives through each other in single file
        and positions never change. With it, passing means finding room.
        """
        fleet = self.fleet
        on_track = fleet.active
        for i, j in self.pairs:
            if not (on_track[i] and on_track[j]):
                continue
            offset = fleet.position[j] - fleet.position[i]
            distance = np.hypot(*offset)
            if distance >= self.CONTACT_DISTANCE or distance < 1e-6:
                continue
            push = offset / distance * (self.CONTACT_DISTANCE - distance) / 2
            fleet.position[i] -= push
            fleet.position[j] += push
            behind = i if fleet.progress[i] < fleet.progress[j] else j
            fleet.speed[behind] *= 0.96

    def _recover(self, i: int):
        """Put a crashed car back on the racing line, facing the right way, and park it briefly."""
        segment = self.fleet.segment[i]
        self.fleet.place(i, self.track.center[segment], self.track.heading[segment], self.fleet.progress[i])
        self.parked_until[i] = self.clock + self.CRASH_PENALTY
        self.slow_since[i] = self.clock + self.CRASH_PENALTY
        self.crashes[i] += 1

    def _count_laps(self):
        completed = np.floor(self.fleet.progress / self.track.length).astype(int)
        for i in np.flatnonzero((completed > self.completed) & ~self.finished):
            lap_time = self.clock - self.lap_started[i]
            self.last_lap[i] = lap_time
            self.best_lap[i] = np.fmin(self.best_lap[i], lap_time)
            self.lap_started[i] = self.clock
            self.completed[i] = completed[i]
            if completed[i] >= self.laps:
                self.finish_time[i] = self.clock

    def standings(self) -> list[int]:
        """Finished cars by time, then everyone else by distance covered."""
        finished = sorted(np.flatnonzero(self.finished), key=lambda i: self.finish_time[i])
        running = sorted(np.flatnonzero(~self.finished), key=lambda i: -self.fleet.progress[i])
        return finished + running

    def current_lap(self, i: int) -> int:
        return int(min(self.completed[i] + 1, self.laps))
