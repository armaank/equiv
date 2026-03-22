import jax
import jax.numpy as jnp
import pytest
from data import generate_toy_data

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "data_type,n",
    [
        ("simple", 6),
        ("structured", 12),
        ("2nd order polynomial", 12),
        ("cosine", 12),
        ("10th order polynomial", 12),
        ("noise", 12),
    ],
)
def test_output_shapes(data_type, n):
    x, y = generate_toy_data(data_type, n, [-1.0, 1.0], seed=0)
    assert x.shape == (n,)
    assert y.shape == (n,)


@pytest.mark.parametrize(
    "data_type",
    ["simple", "structured", "2nd order polynomial", "cosine", "10th order polynomial"],
)
def test_x_is_sorted(data_type):
    x, _ = generate_toy_data(data_type, 20, [0.0, 1.0], seed=0)
    assert jnp.all(x[1:] >= x[:-1])


def test_x_within_bounds():
    lo, hi = -2.0, 2.0
    x, _ = generate_toy_data("noise", 50, [lo, hi], seed=0)
    assert float(jnp.min(x)) >= lo
    assert float(jnp.max(x)) <= hi


def test_invalid_data_type_raises():
    with pytest.raises(ValueError):
        generate_toy_data("bad", 5, [0.0, 1.0], seed=0)
