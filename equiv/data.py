"""Utilities for generating sample data"""

from typing import List, Tuple

import jax
import jax.numpy as jnp
from loguru import logger
from utils import set_seed


def generate_toy_data(
    data_type: str, n_samples: int, bounds: List[float], seed: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate synthetic 1-D regression data.

    Parameters
    ----------
    data_type : {"simple", "structured", "noise"}
        ``"simple"``     – y = sin(x) cos(x²)
        ``"structured"`` – y = x + cos(πx)
        ``"noise"``      – y ~ N(0, 1).
    n_samples : int
        Number of data points.
    bounds : list of float
        [low, high] range for uniform x samples.
    seed : int
        Random seed

    Returns
    -------
    x : jnp.ndarray, shape (n_samples,)
        Input values.
    y : jnp.ndarray, shape (n_samples,)
        Target values.
    """

    assert data_type in ["simple", "structured", "noise"]

    key = set_seed(seed)
    keys = jax.random.split(key, 2)

    x = jax.random.uniform(keys[0], (n_samples,), minval=bounds[0], maxval=bounds[1])

    if data_type == "simple":
        logger.info("Generating simple data")
        x = jnp.sort(x)
        y = jnp.sin(x) * jnp.cos(x**2)
    elif data_type == "structured":
        logger.info("Generating structured data")
        x = jnp.sort(x)
        y = x + jnp.cos(jnp.pi * x)
    else:
        logger.info("Generating randn data")
        y = jax.random.normal(keys[1], (n_samples,))

    # add option to add noise?

    return x, y


def main():
    pass


if __name__ == "__main__":
    main()

    pass
