import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from data import generate_toy_data
from loguru import logger
from ml import fit_polynomial, predict_polynomial

jax.config.update("jax_enable_x64", True)


def make_odr_figure() -> None:
    degree = 150
    seed = 1111
    panels = [
        ("simple", "(a) Simple Data", [-0.5, 0.5], (-1.0, 1.0), 1e-7, 6),
        ("structured", "(b) Structured Data", [1.0, 3.0], (-0.5, 4.0), 1e-7, 12),
        ("noise", "(c) Noisy Data", [-2.0, 2.0], (-3.0, 3.0), 1e-7, 12),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

    for ii, (data_type, title, bounds, y_limits, alpha_reg, n_samples) in enumerate(
        panels
    ):
        logger.info(f"Generating {data_type} data")
        x, y = generate_toy_data(data_type, n_samples, bounds, seed + ii)

        coeffs_unreg, xmin_u, xmax_u = fit_polynomial(
            x, y, degree, alpha=0.0, odr=False
        )
        coeffs_reg, xmin_r, xmax_r = fit_polynomial(
            x, y, degree, alpha=alpha_reg, odr=True
        )

        x_plot_full = jnp.linspace(bounds[0], bounds[1], 400)

        y_unreg = predict_polynomial(x_plot_full, coeffs_unreg, xmin_u, xmax_u)
        y_reg = predict_polynomial(x_plot_full, coeffs_reg, xmin_r, xmax_r)

        ax = axes[ii]
        ax.plot(x, y, "og", ms=4, markeredgecolor="black")

        ax.plot(x_plot_full, y_unreg, color="red", linewidth=2, label="Unregularized")
        ax.plot(
            x_plot_full,
            y_reg,
            color="blue",
            linewidth=2,
            label="Order Dependent Regularization",
        )
        ax.set_title(title)
        ax.set_ylim(*y_limits)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig("odr_sample.png", dpi=300, bbox_inches="tight")


def main():
    make_odr_figure()

    pass


if __name__ == "__main__":
    main()

    pass
