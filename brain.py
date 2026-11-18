"""A small fully connected network, stored as one flat vector per car.

Keeping each brain as a flat "genome" makes evolution simple (crossover and
mutation are just array operations), and a whole generation can be evaluated
in one batched forward pass.
"""
import json
from pathlib import Path

import numpy as np

from sensors import INPUT_COUNT

LAYERS = (INPUT_COUNT, 16, 2)  # outputs: steering, throttle


def genome_size(layers=LAYERS) -> int:
    return sum(n_in * n_out + n_out for n_in, n_out in zip(layers[:-1], layers[1:]))


def random_genomes(count: int, rng: np.random.Generator, layers=LAYERS) -> np.ndarray:
    return rng.normal(0.0, 0.5, size=(count, genome_size(layers)))


def forward(genomes: np.ndarray, inputs: np.ndarray, layers=LAYERS) -> np.ndarray:
    """Run every car's network on its own inputs. tanh keeps outputs in [-1, 1]."""
    count = len(genomes)
    values = inputs
    offset = 0
    for n_in, n_out in zip(layers[:-1], layers[1:]):
        weights = genomes[:, offset:offset + n_in * n_out].reshape(count, n_in, n_out)
        offset += n_in * n_out
        bias = genomes[:, offset:offset + n_out]
        offset += n_out
        values = np.tanh(np.einsum("ci,cio->co", values, weights) + bias)
    return values


def save(path: Path, genome: np.ndarray, **info):
    # .npz instead of pickle: plain arrays, safe to load, and it survives code changes.
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, genome=genome, layers=np.array(LAYERS), info=json.dumps(info))


def load(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path) as data:
        if tuple(data["layers"]) != LAYERS:
            raise ValueError(f"{path.name} was trained with a different network shape")
        return data["genome"].copy(), json.loads(str(data["info"]))
