import evolution
from track import Track


def test_a_few_generations_learn_to_lap_the_circle(tmp_path, monkeypatch):
    monkeypatch.setattr(evolution, "CHAMPION_PATH", tmp_path / "champion.npz")
    monkeypatch.setattr(evolution, "SNAPSHOT_DIR", tmp_path / "snapshots")
    trainer = evolution.Trainer([Track.load("circle")], fresh=True, seed=3)
    trainer.run_headless(8, log=lambda line: None)

    assert any(result.finishers > 0 for result in trainer.history)
    assert (tmp_path / "champion.npz").exists()
    assert (tmp_path / "snapshots" / "gen_005.npz").exists()
