import os

import jax
import matplotlib.pyplot as plt
from loguru import logger


from data import generate_toy_data
from utils import set_seed


def plot_functions():
    logger.info("Setting random seed for reproducibility")

    jax_key = set_seed(1111)
    key_x, key_y = jax.random.split(jax_key)

    logger.info("Generating synthetic data")
    # x, y = generate_toy_data('simple', 5, [-1, 1], jax_key)
    # x, y = generate_toy_data('structured', 8, [0, 4], jax_key)
    x, y = generate_toy_data("noise", 50, [-2, 2], [key_x, key_y])

    fig, ax = plt.subplots()
    ax.plot(x, y, "og", ms=4, markeredgecolor="black", label="Samples")
    plt.xlabel("x")
    plt.ylabel("y", rotation=0)
    plt.title("Synthetic Data")
    plt.show()


def main():
    plot_functions()

    pass


if __name__ == "__main__":
    main()

    pass
