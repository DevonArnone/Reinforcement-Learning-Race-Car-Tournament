from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRACKS_DIR = ROOT / "tracks"
BRAINS_DIR = ROOT / "brains"          # your own training runs (not committed)
PRETRAINED_DIR = ROOT / "pretrained"  # drivers that ship with the repo
SCREENSHOT_DIR = ROOT / "screenshots"

# Window
SCREEN_W, SCREEN_H = 1280, 720
FPS = 60
DT = 1 / FPS

# Track geometry
SAMPLE_SPACING = 8.0       # px between resampled centerline points
SEARCH_WINDOW = 12         # how many points either side of a car's last position to search

# Car
CAR_LENGTH = 30
CAR_WIDTH = 15
MAX_SPEED = 420.0          # px/s
ACCELERATION = 320.0       # px/s^2 at full throttle
BRAKING = 650.0            # px/s^2 at full brake
ROLLING_DRAG = 0.35        # fraction of speed lost per second when coasting
MAX_TURN_RATE = 3.2        # rad/s at full lock
FULL_STEER_SPEED = 140.0   # below this speed steering is proportionally weaker
HIGH_SPEED_GRIP_LOSS = 0.4 # at max speed the turn rate drops by this fraction
