"""Utilities for generating sample data"""

from typing import List, Tuple

import jax
import jax.numpy as jnp
from loguru import logger


def generate_toy_data(
    data_type: str, n_samples: int, bounds: List[int], keys: List[jax.random.PRNGKey]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """generate toy data for order-dependent regularization problem"""

    assert data_type in ["simple", "structured", "noise"]

    x = jax.random.uniform(keys[0], (n_samples, 1), minval=bounds[0], maxval=bounds[1])

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
        y = jax.random.normal(keys[1], (n_samples, 1))

    # add option to add noise?

    return x, y


def main():
    pass


if __name__ == "__main__":
    main()

    pass
