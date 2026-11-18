"""Neuroevolution: run a generation, score it, breed the next one."""
from dataclasses import dataclass, field

import numpy as np

import brain
import config
import sensors
from car import Fleet
from track import Track

CHAMPION_PATH = config.BRAINS_DIR / "champion.npz"
SNAPSHOT_DIR = config.BRAINS_DIR / "snapshots"


@dataclass
class Settings:
    population: int = 40
    elites: int = 4
    tournament: int = 3
    mutation_rate: float = 0.15
    sigma_start: float = 0.4
    sigma_min: float = 0.03
    sigma_decay: float = 0.97
    sensor_noise: float = 0.02
    snapshot_every: int = 5


class Population:
    def __init__(self, settings: Settings, rng: np.random.Generator, seed: np.ndarray | None = None, generation: int = 1):
        self.settings = settings
        self.rng = rng
        self.generation = generation
        if seed is None:
            self.genomes = brain.random_genomes(settings.population, rng)
        else:
            self.genomes = np.repeat(seed[None], settings.population, axis=0)
            self.genomes[1:] = self._mutate(self.genomes[1:])

    @property
    def sigma(self) -> float:
        s = self.settings
        return max(s.sigma_min, s.sigma_start * s.sigma_decay ** (self.generation - 1))

    def _mutate(self, genomes: np.ndarray) -> np.ndarray:
        mask = self.rng.random(genomes.shape) < self.settings.mutation_rate
        return genomes + mask * self.rng.normal(0, self.sigma, genomes.shape)

    def _pick_parent(self, ranked: np.ndarray) -> np.ndarray:
        # Tournament on rank: lower rank index = fitter.
        contenders = self.rng.integers(0, len(ranked), self.settings.tournament)
        return self.genomes[ranked[contenders.min()]]

    def evolve(self, fitness: np.ndarray):
        ranked = np.argsort(-fitness)
        s = self.settings
        children = [self.genomes[i] for i in ranked[:s.elites]]
        while len(children) < s.population:
            mum, dad = self._pick_parent(ranked), self._pick_parent(ranked)
            child = np.where(self.rng.random(mum.shape) < 0.5, mum, dad)
            children.append(self._mutate(child[None])[0])
        self.genomes = np.array(children)
        self.generation += 1


class Episode:
    """One generation driving one lap of a track."""

    STALL_SECONDS = 2.0
    STALL_DISTANCE = 30.0

    def __init__(self, track: Track, genomes: np.ndarray, rng: np.random.Generator,
                 start_along: float = 0.0, sensor_noise: float = 0.0):
        self.track = track
        self.genomes = genomes
        self.rng = rng
        self.sensor_noise = sensor_noise
        self.fleet = Fleet(track, len(genomes))
        self.fleet.place_all_at(start_along)
        self.time = 0.0
        self.time_limit = track.length / 120 + 5
        n = len(genomes)
        self.crashed = np.zeros(n, dtype=bool)
        self.finish_time = np.full(n, np.nan)
        self.best_progress = np.zeros(n)
        self.last_improved = np.zeros(n)
        self.rays = np.zeros((n, sensors.RAY_COUNT))

    @property
    def finished(self) -> np.ndarray:
        return ~np.isnan(self.finish_time)

    @property
    def done(self) -> bool:
        return self.time >= self.time_limit or not self.fleet.active.any()

    def step(self):
        fleet = self.fleet
        self.rays = sensors.cast_rays(fleet)
        inputs = sensors.observe(fleet, self.rays)
        if self.sensor_noise:
            inputs = inputs + self.rng.normal(0, self.sensor_noise, inputs.shape)
        outputs = brain.forward(self.genomes, inputs)
        fleet.drive(outputs[:, 0], outputs[:, 1])
        self.time += config.DT

        crashed = fleet.off_track
        self.crashed |= crashed

        improved = fleet.progress > self.best_progress + self.STALL_DISTANCE
        self.best_progress = np.where(improved, fleet.progress, self.best_progress)
        self.last_improved = np.where(improved, self.time, self.last_improved)
        stalled = self.time - self.last_improved > self.STALL_SECONDS

        just_finished = fleet.active & (fleet.progress >= self.track.length)
        self.finish_time[just_finished] = self.time
        fleet.active &= ~(crashed | stalled | just_finished)

    def fitness(self) -> np.ndarray:
        laps = np.clip(self.fleet.progress / self.track.length, -1, 1)
        speed_bonus = np.where(self.finished, 1 - self.finish_time / self.time_limit, 0)
        return laps + speed_bonus - 0.02 * self.crashed


@dataclass
class GenerationResult:
    generation: int
    track: str
    best_fitness: float
    mean_fitness: float
    finishers: int
    best_lap: float | None


@dataclass
class Trainer:
    tracks: list[Track]
    settings: Settings = field(default_factory=Settings)
    fresh: bool = False
    seed: int | None = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        champion, generation = None, 1
        if self.fresh:
            for old in SNAPSHOT_DIR.glob("*.npz"):
                old.unlink()
        elif CHAMPION_PATH.exists():
            champion, info = brain.load(CHAMPION_PATH)
            generation = info.get("generation", 0) + 1
        self.population = Population(self.settings, self.rng, champion, generation)
        self.best_laps: dict[str, float] = {}
        self.history: list[GenerationResult] = []
        self.episode = self._new_episode()

    def _new_episode(self) -> Episode:
        track = self.tracks[(self.population.generation - 1) % len(self.tracks)]
        # Varying the start point stops brains memorising one sequence of turns.
        start = 0.0 if len(self.history) < 3 else self.rng.uniform(0, track.length)
        return Episode(track, self.population.genomes, self.rng, start, self.settings.sensor_noise)

    def step(self) -> GenerationResult | None:
        """Advance one tick. Returns a result when a generation has just ended."""
        self.episode.step()
        if not self.episode.done:
            return None
        return self._end_generation()

    def _end_generation(self) -> GenerationResult:
        episode = self.episode
        fitness = episode.fitness()
        best = int(fitness.argmax())
        track_name = episode.track.name
        lap = float(np.nanmin(episode.finish_time)) if episode.finished.any() else None
        if lap is not None and lap < self.best_laps.get(track_name, np.inf):
            self.best_laps[track_name] = lap

        result = GenerationResult(self.population.generation, track_name, float(fitness[best]),
                                  float(fitness.mean()), int(episode.finished.sum()), lap)
        self.history.append(result)
        self.save(episode.genomes[best], fitness[best])
        self.population.evolve(fitness)
        self.episode = self._new_episode()
        return result

    def save(self, genome: np.ndarray, fitness: float):
        generation = self.population.generation
        info = {"generation": generation, "fitness": float(fitness)}
        brain.save(CHAMPION_PATH, genome, **info)
        if generation % self.settings.snapshot_every == 0:
            brain.save(SNAPSHOT_DIR / f"gen_{generation:03d}.npz", genome, **info)

    def run_headless(self, generations: int, log=print):
        for _ in range(generations):
            result = None
            while result is None:
                result = self.step()
            log(f"gen {result.generation:3d}  {result.track:<12} best {result.best_fitness:5.2f}  "
                f"mean {result.mean_fitness:5.2f}  finished {result.finishers:2d}  "
                f"lap {'--' if result.best_lap is None else f'{result.best_lap:.2f}s'}")
