"""utilities"""

import random

import jax
import numpy as np


def set_seed(seed: int) -> jax.random.PRNGKey:
    """set random seed for reproducibility"""

    random.seed(seed)
    np.random.seed(seed)

    # jax keys aren't global, requires specific key for each random function call
    jax_key = jax.random.PRNGKey(seed)
    return jax_key


def main():
    pass


if __name__ == "__main__":
    main()

    pass
