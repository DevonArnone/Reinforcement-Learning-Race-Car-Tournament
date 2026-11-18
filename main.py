import argparse
import time

import numpy as np
import pygame

import config
import render
from car import Fleet
from track import Track


class DriveScene:
    """Manual driving, mostly for checking physics and collisions on a new track."""

    def __init__(self, track: Track):
        self.track = track
        surface, origin = render.build_track_surface(track)
        self.camera = render.Camera(track, surface, origin)
        self.fleet = Fleet(track, 1)
        self.sprites = [render.make_car_sprite(render.CAR_COLORS[0])]
        self.reset()

    def reset(self):
        slot = self.track.start_grid(1)[0]
        self.fleet.place(0, slot.position, slot.heading, -slot.distance_behind)
        self.lap_started = None
        self.last_lap = None
        self.best_lap = None
        self.laps_done = 0
        self.clock = 0.0

    def handle(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_o:
            self.camera.toggle_overview()

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
        self.camera.follow(self.fleet.position[0])

    def draw(self, screen):
        screen.fill(render.GRASS)
        self.camera.draw_track(screen)
        render.draw_cars(screen, self.camera, self.fleet, self.sprites)
        current = None if self.lap_started is None else self.clock - self.lap_started
        render.draw_panel(screen, [
            f"{self.track.name}  -  free drive",
            f"Speed   {self.fleet.speed[0]:.0f}",
            f"Lap     {render.format_time(current)}",
            f"Last    {render.format_time(self.last_lap)}",
            f"Best    {render.format_time(self.best_lap)}",
        ])


def save_screenshot(screen):
    config.SCREENSHOT_DIR.mkdir(exist_ok=True)
    path = config.SCREENSHOT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.png"
    pygame.image.save(screen, path)
    print(f"saved {path}")


def run(scene):
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_W, config.SCREEN_H))
    pygame.display.set_caption("RL Race Car Tournament")
    clock = pygame.time.Clock()

    while scene is not None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                scene = None
                break
            if event.type == pygame.KEYDOWN and event.key == pygame.K_F12:
                save_screenshot(screen)
            scene.handle(event)
        if scene is None:
            break
        scene.update(config.DT)
        scene.draw(screen)
        pygame.display.flip()
        clock.tick(config.FPS)
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Train AI drivers and watch them race.")
    parser.add_argument("--track", default="circle", choices=Track.available())
    parser.add_argument("--drive", action="store_true", help="drive a car yourself")
    args = parser.parse_args()

    pygame.init()
    track = Track.load(args.track)
    run(DriveScene(track))


if __name__ == "__main__":
    main()
