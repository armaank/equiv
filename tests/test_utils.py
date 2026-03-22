import jax
import jax.numpy as jnp
import numpy as np
from utils import set_seed

jax.config.update("jax_enable_x64", True)


def test_set_seed_returns_jax_key():
    key = set_seed(0)
    assert isinstance(key, jax.Array)
    assert key.shape == (2,)


def test_set_seed_deterministic():
    k1 = set_seed(42)
    k2 = set_seed(42)
    assert jnp.array_equal(k1, k2)


def test_set_seed_numpy_state():
    set_seed(7)
    a = np.random.rand(3)
    set_seed(7)
    b = np.random.rand(3)
    np.testing.assert_array_equal(a, b)
