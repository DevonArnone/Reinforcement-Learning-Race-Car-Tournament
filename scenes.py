from dataclasses import dataclass

import numpy as np
import pygame

import brain
import config
import race
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
        self.minimap = render.Minimap(track)

    def draw(self, screen, fleet: Fleet, sprites, colors, faded=None, focus: int | None = None):
        screen.fill(render.GRASS)
        self.camera.draw_track(screen)
        render.draw_cars(screen, self.camera, fleet, sprites, faded=faded, on_top=focus)
        if not self.camera.overview:
            self.minimap.draw(screen, fleet.position, colors, focus)


class DriveScene(Scene):
    """Manual driving, mostly for checking physics and collisions on a new track."""

    def __init__(self, track: Track, settings: "MenuSettings | None" = None):
        self.track = track
        self.settings = settings
        self.view = TrackView(track)
        self.fleet = Fleet(track, 1)
        self.color = render.CAR_COLORS[0]
        self.sprites = [render.make_car_sprite(self.color)]
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
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.next_scene = MenuScene(self.settings)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_o:
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
        self.view.draw(screen, self.fleet, self.sprites, [self.color], focus=0)
        current = None if self.lap_started is None else self.clock - self.lap_started
        render.draw_panel(screen, [
            f"{self.track.name}  -  free drive",
            f"Speed   {self.fleet.speed[0]:.0f}",
            f"Lap     {render.format_time(current)}",
            f"Last    {render.format_time(self.last_lap)}",
            f"Best    {render.format_time(self.best_lap)}",
        ])
        render.draw_hint(screen, "Arrows drive   O overview   Esc menu")


class TrainScene(Scene):
    SPEEDS = (1, 4, 16)

    def __init__(self, tracks: list[Track], fresh: bool = False, settings: "MenuSettings | None" = None):
        self.trainer = Trainer(tracks, fresh=fresh)
        self.settings = settings
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
        if event.key == pygame.K_ESCAPE:
            self.next_scene = MenuScene(self.settings)
        elif event.key == pygame.K_f:
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
        lead = self.leader()
        sprites = [self.car_sprite] * fleet.count
        sprites[lead] = self.leader_sprite
        colors = [render.CAR_COLORS[1]] * fleet.count
        colors[lead] = render.CAR_COLORS[2]
        self.view.draw(screen, fleet, sprites, colors, faded=~fleet.active & ~episode.finished, focus=lead)

        if self.show_rays:
            ends = sensors.ray_endpoints(fleet, episode.rays)[lead]
            start = camera.to_screen(fleet.position[lead])
            for end in ends:
                pygame.draw.line(screen, (250, 220, 80), start, camera.to_screen(end), 1)

        trainer = self.trainer
        heat = f"   heat {trainer.heat + 1}/{len(trainer.tracks)}" if len(trainer.tracks) > 1 else ""
        render.draw_panel(screen, [
            f"Training on {episode.track.name}{heat}",
            f"Generation  {trainer.population.generation}",
            f"Alive       {int(fleet.active.sum())} / {fleet.count}",
            f"Lap         {min(int(fleet.progress.max() // episode.track.length) + 1, episode.laps)} / {episode.laps}",
            f"Time        {episode.time:.1f}s",
            f"Best lap    {render.format_time(trainer.best_laps.get(episode.track.name))}",
            f"Mutation    {trainer.population.sigma:.3f}",
            f"Speed       {self.SPEEDS[self.speed_index]}x",
        ], width=260)
        render.draw_fitness_graph(screen, self.trainer.history)
        render.draw_hint(screen, "F fast-forward   R rays   O overview   Esc menu (progress is saved)")


class RaceScene(Scene):
    def __init__(self, track: Track, laps: int, settings: "MenuSettings"):
        self.track = track
        self.laps = laps
        self.settings = settings
        self.view = TrackView(track)
        drivers = race.load_drivers(render.CAR_COLORS)
        self.race = race.Race(track, drivers, laps)
        self.colors = [d.color for d in drivers]
        self.sprites = [render.make_car_sprite(c) for c in self.colors]
        self.focus = None  # None means follow the leader

    def handle(self, event):
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.next_scene = MenuScene(self.settings)
        elif event.key == pygame.K_TAB:
            standings = self.race.standings()
            current = standings.index(self.focus) if self.focus is not None else 0
            self.focus = standings[(current + 1) % len(standings)]
        elif event.key == pygame.K_l:
            self.focus = None
        elif event.key == pygame.K_o:
            self.view.camera.toggle_overview()
        elif self.race.over and event.key == pygame.K_r:
            self.next_scene = RaceScene(self.track, self.laps, self.settings)
        elif self.race.over and event.key == pygame.K_RETURN:
            self.next_scene = MenuScene(self.settings)

    def target(self) -> int:
        return self.race.standings()[0] if self.focus is None else self.focus

    def update(self, dt):
        if not self.race.over:
            self.race.step(dt)
        self.view.camera.follow(self.race.fleet.position[self.target()])

    def draw(self, screen):
        r = self.race
        parked = r.clock < r.parked_until
        self.view.draw(screen, r.fleet, self.sprites, self.colors, faded=parked, focus=self.target())
        self._draw_name_tag(screen, self.target())
        self._draw_leaderboard(screen)

        leader = r.standings()[0]
        render.draw_banner(screen, f"Lap {r.current_lap(leader)} / {r.laps}", y=24, size=34)
        if not r.started:
            render.draw_banner(screen, str(int(np.ceil(r.countdown))), y=config.SCREEN_H / 2 - 60, size=140)
        elif r.clock < 1.0:
            render.draw_banner(screen, "GO", y=config.SCREEN_H / 2 - 60, size=140, color=(120, 230, 120))

        if r.over:
            self._draw_results(screen)
        else:
            render.draw_hint(screen, "Tab next car   L follow leader   O overview   Esc menu")

    def _draw_name_tag(self, screen, i: int):
        position = self.view.camera.to_screen(self.race.fleet.position[i]) - (0, 26)
        label = render.font(20).render(self.race.drivers[i].name, True, render.TEXT)
        screen.blit(label, label.get_rect(center=position))

    def _draw_leaderboard(self, screen):
        r = self.race
        lines = []
        for place, i in enumerate(r.standings(), start=1):
            status = "finished" if r.finished[i] else f"lap {r.current_lap(i)}"
            lines.append(f"{place} {r.drivers[i].name:<10}{status:>8}")
        top = 16 if self.view.camera.overview else self.view.minimap.surface.get_height() + 28
        render.draw_panel(screen, lines, topleft=(config.SCREEN_W - 256, top), size=24, width=240,
                          swatches=[r.drivers[i].color for i in r.standings()], mono=True)

    def _draw_results(self, screen):
        r = self.race
        winner_time = np.nanmin(r.finish_time)
        rows = ["Pos  Driver       Trained    Time         Best lap"]
        for place, i in enumerate(r.standings(), start=1):
            d = r.drivers[i]
            if r.finished[i]:
                gap = "winner" if place == 1 else f"+{r.finish_time[i] - winner_time:.2f}s"
                time = render.format_time(r.finish_time[i]) if place == 1 else gap
            else:
                time = "DNF"
            best = None if np.isnan(r.best_lap[i]) else r.best_lap[i]
            rows.append(f"{place:<4} {d.name:<12} gen {d.generation:<6} {time:<12} {render.format_time(best)}")
        rows += ["", "R race again     Enter menu"]
        render.draw_panel(screen, rows, topleft=(config.SCREEN_W / 2 - 300, 170), size=26, width=600, mono=True)


@dataclass
class MenuSettings:
    track_index: int = 0
    laps: int = 3
    train_all: bool = True
    fresh: bool = False


class MenuScene(Scene):
    ITEMS = ["Watch a race", "Train drivers", "Drive yourself", "Track", "Laps", "Train on", "Training", "Quit"]

    def __init__(self, settings: MenuSettings | None = None):
        self.settings = settings or MenuSettings()
        self.track_names = Track.available()
        self.tracks = {name: Track.load(name) for name in self.track_names}
        self.selected = 0
        self.previews: dict[str, pygame.Surface] = {}
        self.drivers_dir = race.drivers_dir()
        self.champion_generation = None
        if self.drivers_dir is not None:
            self.champion_generation = brain.load(self.drivers_dir / "champion.npz")[1]["generation"]

    @property
    def track(self) -> Track:
        return self.tracks[self.track_names[self.settings.track_index]]

    def value(self, item: str) -> str | None:
        s = self.settings
        return {
            "Track": self.track.name,
            "Laps": str(s.laps),
            "Train on": "all tracks" if s.train_all else "this track",
            "Training": "from scratch" if s.fresh else "continue champion",
        }.get(item)

    def handle(self, event):
        if event.type != pygame.KEYDOWN:
            return
        s = self.settings
        item = self.ITEMS[self.selected]
        if event.key == pygame.K_ESCAPE:
            self.next_scene = False
        elif event.key in (pygame.K_UP, pygame.K_w):
            self.selected = (self.selected - 1) % len(self.ITEMS)
        elif event.key in (pygame.K_DOWN, pygame.K_s):
            self.selected = (self.selected + 1) % len(self.ITEMS)
        elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
            step = 1 if event.key == pygame.K_RIGHT else -1
            if item == "Track":
                s.track_index = (s.track_index + step) % len(self.track_names)
            elif item == "Laps":
                s.laps = int(np.clip(s.laps + step, 1, 10))
            elif item == "Train on":
                s.train_all = not s.train_all
            elif item == "Training":
                s.fresh = not s.fresh
        elif event.key == pygame.K_RETURN:
            self.activate(item)

    def activate(self, item: str):
        s = self.settings
        if item == "Watch a race" and self.drivers_dir is not None:
            self.next_scene = RaceScene(self.track, s.laps, s)
        elif item == "Train drivers":
            tracks = list(self.tracks.values()) if s.train_all else [self.track]
            self.next_scene = TrainScene(tracks, fresh=s.fresh, settings=s)
            s.fresh = False
        elif item == "Drive yourself":
            self.next_scene = DriveScene(self.track, settings=s)
        elif item == "Quit":
            self.next_scene = False

    def draw(self, screen):
        screen.fill((24, 27, 32))
        render.draw_banner(screen, "RL Race Car Tournament", y=70, size=64)
        render.draw_banner(screen, "neuroevolution drivers, raced against each other", y=125, size=26, color=render.MUTED)

        y = 210
        for i, item in enumerate(self.ITEMS):
            selected = i == self.selected
            value = self.value(item)
            text = f"{item}:  < {value} >" if value else item
            color = (250, 220, 80) if selected else render.TEXT
            label = render.font(34).render(("> " if selected else "  ") + text, True, color)
            screen.blit(label, (120, y))
            y += 46

        if self.champion_generation is None:
            status = "No trained drivers yet. Train first."
        else:
            source = "pretrained" if self.drivers_dir == config.PRETRAINED_DIR else "your"
            status = f"Racing {source} drivers - champion trained for {self.champion_generation} generations"
        screen.blit(render.font(24).render(status, True, render.MUTED), (120, y + 20))
        self._draw_preview(screen)
        render.draw_hint(screen, "Up/Down select   Left/Right change   Enter start   Esc quit")

    def _draw_preview(self, screen, box=(460, 360)):
        track = self.track
        if track.name not in self.previews:
            self.previews[track.name] = render.track_thumbnail(track, box)
        preview = self.previews[track.name]
        x = config.SCREEN_W - box[0] - 100
        y = 220 + (box[1] - preview.get_height()) / 2
        screen.blit(preview, (x, y))
        caption = f"{track.name}   {track.length / 1000:.1f}k px lap"
        screen.blit(render.font(24).render(caption, True, render.MUTED), (x, y + preview.get_height() + 10))
