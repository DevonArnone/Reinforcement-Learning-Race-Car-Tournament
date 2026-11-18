import numpy as np

import brain


def test_forward_gives_one_output_pair_per_car():
    rng = np.random.default_rng(0)
    genomes = brain.random_genomes(5, rng)
    outputs = brain.forward(genomes, rng.normal(size=(5, brain.LAYERS[0])))
    assert outputs.shape == (5, 2)
    assert np.all(np.abs(outputs) <= 1)


def test_save_and_load_round_trip(tmp_path):
    genome = brain.random_genomes(1, np.random.default_rng(1))[0]
    path = tmp_path / "car.npz"
    brain.save(path, genome, generation=12)
    loaded, info = brain.load(path)
    assert np.array_equal(loaded, genome)
    assert info["generation"] == 12
