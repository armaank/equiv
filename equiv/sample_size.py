"""sample_size

sample size experiments
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from data import generate_toy_data
from loguru import logger
from ml import fit_polynomial, predict_polynomial
from utils import set_seed

jax.config.update("jax_enable_x64", True)


def _eval_rmse(
    x_test: jnp.ndarray,
    y_test: jnp.ndarray,
    coeffs: jnp.ndarray,
    x_min: float,
    x_max: float,
) -> float:
    """Evaluate root mean squared error of a fitted polynomial on a test set."""
    y_pred = predict_polynomial(x_test, coeffs, x_min, x_max)
    return float(jnp.sqrt(jnp.mean((y_pred - y_test) ** 2)))


def run_sample_size_experiment(
    data_type: str,
    sample_sizes: list[int],
    alpha: float,
    n_trials: int = 100,
    noise_sigma: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    """Run a sample size experiment comparing degree-2, degree-10, and regularized degree-10 polynomials.

    For each sample size and trial, generates noisy training data, fits three models,
    and records test RMSE. Returns per-sample-size mean and std of RMSE across trials.

    Parameters
    ----------
    data_type : str
        Data generating process passed to `generate_toy_data`.
    sample_sizes : list of int
        Training set sizes to evaluate.
    alpha : float
        Regularization strength for the degree-10 soft-ODR model.
    n_trials : int
        Number of independent trials per sample size.
    noise_sigma : float
        Standard deviation of Gaussian noise added to training targets.
    seed : int
        Base random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``n``, ``degree_2_mean``, ``degree_2_std``,
        ``degree_10_mean``, ``degree_10_std``,
        ``degree_10_reg_mean``, ``degree_10_reg_std``.
        One row per entry in ``sample_sizes``.
    """
    key = set_seed(seed)
    x_test, y_test = generate_toy_data(data_type, 400, [0.0, 1.0], seed=seed + 9999)

    records = []
    for i, n in enumerate(sample_sizes):
        for trial in range(n_trials):
            x_train, y_noiseless = generate_toy_data(
                data_type, n, [0.0, 1.0], seed=seed + i * n_trials + trial
            )
            key, noise_key = jax.random.split(key)
            y_train = y_noiseless + noise_sigma * jax.random.normal(noise_key, (n,))

            coeffs_deg2, xmin_deg2, xmax_deg2 = fit_polynomial(
                x_train, y_train, degree=2, alpha=0.0, odr=False
            )
            coeffs_deg10, xmin_deg10, xmax_deg10 = fit_polynomial(
                x_train, y_train, degree=10, alpha=0.0, odr=False
            )
            coeffs_reg, xmin_reg, xmax_reg = fit_polynomial(
                x_train, y_train, degree=10, alpha=alpha, odr="quad_odr"
            )

            records.append(
                {
                    "n": n,
                    "degree_2": _eval_rmse(
                        x_test, y_test, coeffs_deg2, xmin_deg2, xmax_deg2
                    ),
                    "degree_10": _eval_rmse(
                        x_test, y_test, coeffs_deg10, xmin_deg10, xmax_deg10
                    ),
                    "degree_10_reg": _eval_rmse(
                        x_test, y_test, coeffs_reg, xmin_reg, xmax_reg
                    ),
                }
            )

    agg = (
        pd.DataFrame(records)
        .groupby("n")[["degree_2", "degree_10", "degree_10_reg"]]
        .agg(["mean", "std"])
    )
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    return agg.reset_index()


def make_sample_size_figure() -> None:
    """Generate and save the sample size experiment figure (3 panels, one per data type).

    Runs `run_sample_size_experiment` for the degree-2 polynomial, 10th-order polynomial,
    and cosine targets, then plots test RMSE vs. training set size on a log scale with
    ±1 std shaded bands. Saves the figure to ``sample_size.png``.
    """
    sample_sizes = list(range(10, 101, 5))

    results_deg2 = run_sample_size_experiment(
        "2nd order polynomial", sample_sizes, alpha=0.01, seed=1
    )
    results_cosine = run_sample_size_experiment(
        "cosine", sample_sizes, alpha=0.01, seed=2
    )
    results_deg10 = run_sample_size_experiment(
        "10th order polynomial", sample_sizes, alpha=0.001, seed=3
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    def plot_panel(ax, results, title):
        for model, color, label in [
            ("degree_2", "blue", "Degree 2"),
            ("degree_10_reg", "green", "Degree 10 + Reg"),
            ("degree_10", "purple", "Degree 10"),
        ]:
            mean = results[f"{model}_mean"]
            std = results[f"{model}_std"]
            ax.semilogy(results["n"], mean, color=color, linewidth=2, label=label)
            ax.fill_between(
                results["n"], mean - std, mean + std, color=color, alpha=0.2
            )
        ax.set_title(title)
        ax.set_xlabel("# Training Samples")
        ax.set_ylabel("Test RMSE")
        ax.grid(True, alpha=0.3)

    plot_panel(axes[0], results_deg2, "Degree 2 target")
    plot_panel(axes[1], results_deg10, "Degree 10 target")
    plot_panel(axes[2], results_cosine, "Cosine target")

    axes[0].legend(loc="best", fontsize=9)
    fig.tight_layout()
    logger.info("Saving figure to 'sample_size.png'")
    fig.savefig("sample_size.png", dpi=300, bbox_inches="tight")


def main():
    make_sample_size_figure()


if __name__ == "__main__":
    main()
