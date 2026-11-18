import numpy as np
import pytest

from car import Fleet
from track import Track


@pytest.fixture(params=Track.available())
def track(request):
    return Track.load(request.param)


def test_centerline_is_evenly_spaced(track):
    spacing = track.segment_length
    assert spacing.max() - spacing.min() < 0.5


def test_locate_finds_points_on_the_centerline(track):
    indices = np.arange(0, track.count, 7)
    segment, along, gap = track.locate(track.center[indices], indices)
    assert np.allclose(gap, 0, atol=1e-6)
    difference = np.abs(along - track.distance[indices])
    assert np.allclose(np.minimum(difference, track.length - difference), 0, atol=1e-6)


def test_progress_increases_when_driving_the_centerline(track):
    fleet = Fleet(track, 1)
    fleet.place(0, track.center[0], track.heading[0])
    previous = fleet.progress[0]
    for point in track.center[1:]:
        fleet.position[0] = point
        fleet._update_progress(np.array([True]))
        assert fleet.progress[0] > previous
        previous = fleet.progress[0]


def test_driving_backwards_over_the_line_is_not_a_lap(track):
    fleet = Fleet(track, 1)
    fleet.place(0, track.center[0], track.heading[0])
    for point in track.center[::-1][:20]:
        fleet.position[0] = point
        fleet._update_progress(np.array([True]))
    assert fleet.laps[0] == -1
    assert fleet.progress[0] < 0


def test_start_grid_is_on_the_road_and_behind_the_line(track):
    slots = track.start_grid(8)
    positions = np.array([s.position for s in slots])
    hints = np.array([track.nearest_index(p) for p in positions])
    _, along, gap = track.locate(positions, hints)
    assert np.all(gap < track.width / 2)
    assert np.all(along > track.length / 2)


def test_corners_are_wider_than_the_road(track):
    # A corner tighter than half the road width would fold the inside edge over itself.
    turn = np.abs(np.diff(np.unwrap(track.heading)))
    radius = track.segment_length[:-1] / np.maximum(turn, 1e-9)
    assert radius.min() > track.width / 2 + 10


def test_road_does_not_overlap_itself(track):
    # Any two points that are far apart along the lap must also be far apart in
    # space, otherwise the road surfaces would touch.
    points = track.center[::3]
    distance = track.distance[::3]
    gaps = np.linalg.norm(points[:, None] - points[None], axis=-1)
    along = np.abs(distance[:, None] - distance[None])
    along = np.minimum(along, track.length - along)
    far_along = along > track.width * 2
    assert gaps[far_along].min() > track.width * 1.2
