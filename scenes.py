import numpy as np
import pygame

import render
import sensors
from car import Fleet
from evolution import Trainer
from track import Track


class Scene:
    next_scene = None  # set to a Scene to switch, or to False to quit

    def handle(self, event): ...
    def update(self, dt): ...
    def draw(self, screen): ...


class TrackView:
    """Pre-rendered track plus a camera, cached per track."""

    _cache: dict[str, tuple[pygame.Surface, np.ndarray]] = {}

    def __init__(self, track: Track):
        if track.name not in self._cache:
            self._cache[track.name] = render.build_track_surface(track)
        surface, origin = self._cache[track.name]
        self.camera = render.Camera(track, surface, origin)


class DriveScene(Scene):
    """Manual driving, mostly for checking physics and collisions on a new track."""

    def __init__(self, track: Track):
        self.track = track
        self.view = TrackView(track)
        self.fleet = Fleet(track, 1)
        self.sprites = [render.make_car_sprite(render.CAR_COLORS[0])]
        self.reset()

    def reset(self):
        slot = self.track.start_grid(1)[0]
        self.fleet.place(0, slot.position, slot.heading, -slot.distance_behind)
        self.clock = 0.0
        self.lap_started = None
        self.last_lap = None
        self.best_lap = None
        self.laps_done = 0

    def handle(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_o:
            self.view.camera.toggle_overview()

    def update(self, dt):
        keys = pygame.key.get_pressed()
        steer = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        throttle = keys[pygame.K_UP] - keys[pygame.K_DOWN]
        self.fleet.drive(np.array([steer]), np.array([throttle]), dt)
        self.clock += dt

        if self.fleet.off_track[0]:
            self.reset()
            return

        laps = int(self.fleet.laps[0]) + 1  # +1 because the car starts behind the line
        if laps > self.laps_done:
            if self.lap_started is not None:
                self.last_lap = self.clock - self.lap_started
                self.best_lap = min(self.best_lap or self.last_lap, self.last_lap)
            self.lap_started = self.clock
            self.laps_done = laps
        self.view.camera.follow(self.fleet.position[0])

    def draw(self, screen):
        screen.fill(render.GRASS)
        self.view.camera.draw_track(screen)
        render.draw_cars(screen, self.view.camera, self.fleet, self.sprites)
        current = None if self.lap_started is None else self.clock - self.lap_started
        render.draw_panel(screen, [
            f"{self.track.name}  -  free drive",
            f"Speed   {self.fleet.speed[0]:.0f}",
            f"Lap     {render.format_time(current)}",
            f"Last    {render.format_time(self.last_lap)}",
            f"Best    {render.format_time(self.best_lap)}",
        ])


class TrainScene(Scene):
    SPEEDS = (1, 4, 16)

    def __init__(self, tracks: list[Track], fresh: bool = False):
        self.trainer = Trainer(tracks, fresh=fresh)
        self.speed_index = 0
        self.show_rays = True
        self.car_sprite = render.make_car_sprite(render.CAR_COLORS[1])
        self.leader_sprite = render.make_car_sprite(render.CAR_COLORS[2])
        self.views: dict[str, TrackView] = {}

    @property
    def view(self) -> TrackView:
        track = self.trainer.episode.track
        if track.name not in self.views:
            self.views[track.name] = TrackView(track)
        return self.views[track.name]

    def handle(self, event):
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_f:
            self.speed_index = (self.speed_index + 1) % len(self.SPEEDS)
        elif event.key == pygame.K_r:
            self.show_rays = not self.show_rays
        elif event.key == pygame.K_o:
            self.view.camera.toggle_overview()

    def update(self, dt):
        for _ in range(self.SPEEDS[self.speed_index]):
            self.trainer.step()
        episode = self.trainer.episode
        self.view.camera.follow(episode.fleet.position[self.leader()])

    def leader(self) -> int:
        fleet = self.trainer.episode.fleet
        progress = np.where(fleet.active, fleet.progress, -np.inf)
        return int(progress.argmax()) if fleet.active.any() else int(fleet.progress.argmax())

    def draw(self, screen):
        episode = self.trainer.episode
        fleet = episode.fleet
        camera = self.view.camera
        screen.fill(render.GRASS)
        camera.draw_track(screen)

        lead = self.leader()
        if self.show_rays:
            ends = sensors.ray_endpoints(fleet, episode.rays)[lead]
            start = camera.to_screen(fleet.position[lead])
            for end in ends:
                pygame.draw.line(screen, (250, 220, 80), start, camera.to_screen(end), 1)

        sprites = [self.car_sprite] * fleet.count
        sprites[lead] = self.leader_sprite
        render.draw_cars(screen, camera, fleet, sprites, faded=~fleet.active & ~episode.finished, on_top=lead)

        population = self.trainer.population
        best_lap = self.trainer.best_laps.get(episode.track.name)
        render.draw_panel(screen, [
            f"Training on {episode.track.name}",
            f"Generation  {population.generation}",
            f"Alive       {int(fleet.active.sum())} / {fleet.count}",
            f"Finished    {int(episode.finished.sum())}",
            f"Time        {episode.time:.1f}s",
            f"Best lap    {render.format_time(best_lap)}",
            f"Mutation    {population.sigma:.3f}",
            f"Speed       {self.SPEEDS[self.speed_index]}x",
        ], width=230)
        render.draw_fitness_graph(screen, self.trainer.history)
        render.draw_hint(screen, "F fast-forward   R rays   O overview   Esc back")
