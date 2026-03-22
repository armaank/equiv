import jax
from odr import make_odr_figure
from sample_size import make_sample_size_figure

jax.config.update("jax_enable_x64", True)


def main():
    make_odr_figure()
    make_sample_size_figure()


if __name__ == "__main__":
    main()
