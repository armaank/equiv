"""Utilities for generating sample data"""

import jax
import jax.numpy as jnp
from utils import set_seed


def generate_toy_data(
    data_type: str, n_samples: int, bounds: list[float], seed: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate synthetic 1-D regression data.

    Parameters
    ----------
    data_type : {"simple", "structured", "2nd order polynomial", "cosine", "10th order polynomial", "noise"}
        ``"simple"``                – y = sin(x) cos(x²)
        ``"structured"``            – y = x + cos(πx)
        ``"2nd order polynomial"``  – y = x²
        ``"cosine"``                – y = cos(3/2 π x)
        ``"10th order polynomial"`` – y = -36x + 49x⁵ - 14x⁷ + x¹⁰
        ``"noise"``                 – y ~ N(0, 1)
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
    valid = {
        "simple",
        "structured",
        "2nd order polynomial",
        "cosine",
        "10th order polynomial",
        "noise",
    }
    if data_type not in valid:
        raise ValueError(
            f"Unknown data_type {data_type!r}. Must be one of {sorted(valid)}"
        )

    key = set_seed(seed)
    keys = jax.random.split(key, 2)

    x = jax.random.uniform(keys[0], (n_samples,), minval=bounds[0], maxval=bounds[1])

    if data_type == "simple":
        x = jnp.sort(x)
        y = jnp.sin(x) * jnp.cos(x**2)
    elif data_type == "structured":
        x = jnp.sort(x)
        y = x + jnp.cos(jnp.pi * x)
    elif data_type == "2nd order polynomial":
        x = jnp.sort(x)
        y = x**2
    elif data_type == "cosine":
        x = jnp.sort(x)
        y = jnp.cos(3 / 2 * jnp.pi * x)
    # one of the paper claims to fit a 15th order polynomial, but other papers provide
    # formulas for only 10th order polynomials...
    elif data_type == "10th order polynomial":
        x = jnp.sort(x)
        y = -36 * x + 49 * x**5 - 14 * x**7 + x**10

    else:
        y = jax.random.normal(keys[1], (n_samples,))

    return x, y


if __name__ == "__main__":
    pass
