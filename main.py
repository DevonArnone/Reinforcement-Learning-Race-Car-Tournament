import argparse
import time

import pygame

import config
from evolution import Trainer
from scenes import DriveScene, TrainScene
from track import Track


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
        if scene.next_scene is not None:
            scene = scene.next_scene or None
            continue
        scene.draw(screen)
        pygame.display.flip()
        clock.tick(config.FPS)
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Train AI drivers and watch them race.")
    parser.add_argument("--track", default="circle", help="track name, or 'all' to train on every track")
    parser.add_argument("--drive", action="store_true", help="drive a car yourself")
    parser.add_argument("--train", action="store_true", help="train AI drivers")
    parser.add_argument("--fresh", action="store_true", help="start training from scratch")
    parser.add_argument("--headless", action="store_true", help="train without a window")
    parser.add_argument("--gens", type=int, default=50, help="generations to run headless")
    args = parser.parse_args()

    names = Track.available() if args.track == "all" else [args.track]
    tracks = [Track.load(name) for name in names]

    if args.train and args.headless:
        Trainer(tracks, fresh=args.fresh).run_headless(args.gens)
    elif args.train:
        run(TrainScene(tracks, fresh=args.fresh))
    else:
        run(DriveScene(tracks[0]))


if __name__ == "__main__":
    main()
