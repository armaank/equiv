"""ml modules"""

import jax.numpy as jnp


def _scale_x(x: jnp.ndarray) -> tuple[jnp.ndarray, float, float]:
    """Scale x to [-1, 1] using min-max normalization.

    Parameters
    ----------
    x : jnp.ndarray, shape (n,)
        Raw input values.

    Returns
    -------
    x_scaled : jnp.ndarray, shape (n,)
        Values mapped to [-1, 1].
    x_min : float
        Observed minimum of x.
    x_max : float
        Observed maximum of x.
    """
    x_min = float(jnp.min(x))
    x_max = float(jnp.max(x))
    x_scaled = 2.0 * (x - x_min) / (x_max - x_min + 1e-12) - 1.0
    return x_scaled, x_min, x_max


def fit_polynomial(
    x: jnp.ndarray,
    y: jnp.ndarray,
    degree: int,
    alpha: float | None,
    odr: str | bool = False,
):
    """Fit a polynomial by closed-form least squares with optional ODR.

    Parameters
    ----------
    x : jnp.ndarray, shape (n_samples,)
        Input values.
    y : jnp.ndarray, shape (n_samples,)
        Target values.
    degree : int
        Polynomial degree.
    alpha : float or None
        Regularization strength. Required when odr is not False.
    odr : {"exp_odr", "quad_odr", False}
        ``"exp_odr"``  – exponential penalty 2^j per degree j (aggressive).
        ``"quad_odr"`` – quadratic penalty j² per degree j (gentle).
        ``False``      – no order-dependent regularization.

    Returns
    -------
    coeffs : jnp.ndarray, shape (degree + 1,)
        Fitted polynomial coefficients.
    x_min : float
        Observed minimum of x (for rescaling at predict time).
    x_max : float
        Observed maximum of x (for rescaling at predict time).
    """
    # scale input
    x_scaled, x_min, x_max = _scale_x(x)

    # create polynomial feature matrix
    X = jnp.stack([x_scaled**j for j in range(degree + 1)], axis=1)
    X = X.reshape(-1, degree + 1)
    # create regularization matrix

    if odr == "exp_odr":
        # scale penalties by exponentially increasing degree (2**j)
        penalties = 2.0 ** jnp.arange(degree + 1, dtype=float)
        penalties = penalties.at[0].set(0.0)  # do not regularize intercept term
        coeffs = jnp.linalg.solve(
            jnp.matmul(X.T, X) + alpha * jnp.diag(penalties), jnp.matmul(X.T, y)
        )
    elif odr == "quad_odr":
        # scale by quadratic penalty (j**2)
        penalties = jnp.arange(degree + 1, dtype=float) ** 2
        penalties = penalties.at[0].set(0.0)
        coeffs = jnp.linalg.solve(
            jnp.matmul(X.T, X) + alpha * jnp.diag(penalties), jnp.matmul(X.T, y)
        )
    else:
        # solve for coefficients without order-dependent regularization
        coeffs = jnp.linalg.solve(jnp.matmul(X.T, X), jnp.matmul(X.T, y))

    return coeffs, x_min, x_max


def predict_polynomial(
    x: jnp.ndarray, coeffs: jnp.ndarray, x_min: float, x_max: float
) -> jnp.ndarray:
    """Evaluate a fitted polynomial at new input points.

    Parameters
    ----------
    x : jnp.ndarray, shape (m,)
        Input values.
    coeffs : jnp.ndarray, shape (degree + 1,)
        Polynomial coefficients from `fit_polynomial`.
    x_min : float
        Training data minimum (returned by `fit_polynomial`).
    x_max : float
        Training data maximum (returned by `fit_polynomial`).

    Returns
    -------
    jnp.ndarray, shape (m,)
        Predicted values.
    """
    x_scaled = 2.0 * (x - x_min) / (x_max - x_min + 1e-12) - 1.0
    X = jnp.stack([x_scaled**j for j in range(len(coeffs))], axis=1)
    return jnp.matmul(X, coeffs)
