import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml import _scale_x, fit_polynomial, predict_polynomial

jax.config.update("jax_enable_x64", True)


def test_scale_x_range():
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x_scaled, x_min, x_max = _scale_x(x)
    assert float(jnp.min(x_scaled)) == pytest.approx(-1.0, abs=1e-5)
    assert float(jnp.max(x_scaled)) == pytest.approx(1.0, abs=1e-5)
    assert x_min == pytest.approx(1.0)
    assert x_max == pytest.approx(5.0)


def test_fit_polynomial_coeff_shape():
    x = jnp.linspace(0.0, 1.0, 10)
    y = x**2
    coeffs, _, _ = fit_polynomial(x, y, degree=3, alpha=0.0, odr=False)
    assert coeffs.shape == (4,)


def test_fit_polynomial_exp_odr_coeff_shape():
    x = jnp.linspace(0.0, 1.0, 20)
    y = jnp.sin(x)
    coeffs, _, _ = fit_polynomial(x, y, degree=5, alpha=1e-3, odr="exp_odr")
    assert coeffs.shape == (6,)


def test_fit_polynomial_quad_odr_coeff_shape():
    x = jnp.linspace(0.0, 1.0, 20)
    y = jnp.sin(x)
    coeffs, _, _ = fit_polynomial(x, y, degree=5, alpha=1e-3, odr="quad_odr")
    assert coeffs.shape == (6,)


def test_predict_polynomial_shape():
    x = jnp.linspace(0.0, 1.0, 10)
    y = x**2
    coeffs, x_min, x_max = fit_polynomial(x, y, degree=3, alpha=0.0, odr=False)
    y_pred = predict_polynomial(jnp.linspace(0.0, 1.0, 50), coeffs, x_min, x_max)
    assert y_pred.shape == (50,)


def test_predict_interpolates_linear():
    """Test that fitting a degree 1 polynomial to linear data recovers the original function."""
    x = jnp.linspace(0.0, 1.0, 6)
    y = 2.0 * x + 1.0
    coeffs, x_min, x_max = fit_polynomial(x, y, degree=1, alpha=0.0, odr=False)
    y_pred = predict_polynomial(x, coeffs, x_min, x_max)
    np.testing.assert_allclose(np.array(y_pred), np.array(y), atol=1e-5)
