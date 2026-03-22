"""utilities"""

import random

import jax
import numpy as np


def set_seed(seed: int) -> jax.random.PRNGKey:
    """Set random seeds for Python, numpy, and JAX for reproducibility.

    Parameters
    ----------
    seed : int
        Integer seed value.

    Returns
    -------
    jax.random.PRNGKey
        JAX PRNG key initialized with `seed`.
    """

    random.seed(seed)
    np.random.seed(seed)

    # jax keys aren't global, requires specific key for each random function call
    jax_key = jax.random.PRNGKey(seed)
    return jax_key


if __name__ == "__main__":
    pass
