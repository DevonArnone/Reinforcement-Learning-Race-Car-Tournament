import numpy as np

import brain
from race import Driver, Race
from track import Track


def make_race(laps=2, cars=3):
    rng = np.random.default_rng(0)
    drivers = [Driver(f"car{i}", (255, 0, 0), genome, generation=i)
               for i, genome in enumerate(brain.random_genomes(cars, rng))]
    return Race(Track.load("oval"), drivers, laps)


def test_cars_start_behind_the_line_and_nobody_has_a_lap():
    race = make_race()
    assert np.all(race.fleet.progress < 0)
    race._count_laps()
    assert np.all(race.completed == 0)


def test_lap_times_and_finishing_order():
    race = make_race(laps=2)
    length = race.track.length
    race.clock = 10.0
    race.fleet.progress[:] = [length + 1, 0.5 * length, length + 1]
    race._count_laps()
    assert race.last_lap[0] == 10.0 and np.isnan(race.last_lap[1])

    race.clock = 18.0
    race.fleet.progress[2] = 2 * length + 1
    race._count_laps()
    race.clock = 19.0
    race.fleet.progress[0] = 2 * length + 1
    race._count_laps()

    assert race.finished.tolist() == [True, False, True]
    assert race.standings() == [2, 0, 1]
    assert race.best_lap[2] == 8.0


def test_crashed_car_is_put_back_on_the_road_and_parked():
    race = make_race()
    race.countdown = 0
    race.fleet.position[0] += 500  # well off the track
    race._recover(0)
    _, _, gap = race.track.locate(race.fleet.position[:1], race.fleet.segment[:1])
    assert gap[0] < 1e-6
    assert race.parked_until[0] == race.clock + Race.CRASH_PENALTY
