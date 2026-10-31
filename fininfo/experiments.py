"""Synthetic misspecification studies for Section 6 of the research draft.

Run from the repository root::

    python -m fininfo.experiments

Both studies draw from a known generating model and measure the empirical
coverage of nominal 95% predictive intervals for a scalar payoff
``Y = Z + nu``. They test the software against its own assumptions; they say
nothing about real-market predictability. Results, seeds, sample counts and
package versions are written to ``fininfo/results/results.json``.
"""

import json
import platform
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.special import ndtr, ndtri  # noqa: E402

from fininfo.core import condition, reliability_update  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
COVERAGE = 0.95


def _gaussian_coverage(target, mean, variance):
    radius = ndtri((1 + COVERAGE) / 2) * np.sqrt(variance)
    return float(np.mean(np.abs(target - mean) <= radius))


def _mixture_bounds(weights, means, standard_deviations, probability, iterations=200):
    """Vectorized bisection for the quantile of a per-draw Gaussian mixture."""

    lower = (means - 10 * standard_deviations).min(axis=1)
    upper = (means + 10 * standard_deviations).max(axis=1)
    for _ in range(iterations):
        middle = (lower + upper) / 2
        cdf = (weights * ndtr((middle[:, None] - means) / standard_deviations)).sum(axis=1)
        below = cdf < probability
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    return (lower + upper) / 2


def correlated_packets(draws=100_000, seed=0):
    """Two packets with strongly correlated errors; full versus diagonal R."""

    rng = np.random.default_rng(seed)
    prior_mean, prior_variance, payoff_noise = 0.0, 1.0, 0.2
    noise = np.array([[0.5, 0.4], [0.4, 0.5]])
    measurement = np.ones((2, 1))
    state = rng.normal(prior_mean, np.sqrt(prior_variance), draws)
    payoff = state + rng.normal(0.0, np.sqrt(payoff_noise), draws)
    signals = state[:, None] + rng.multivariate_normal(np.zeros(2), noise, draws)

    def predictive(noise_covariance):
        update = jax.vmap(lambda s: condition(jnp.array([prior_mean]), jnp.array([[prior_variance]]),
                                              measurement, s, noise_covariance))(jnp.asarray(signals))
        return np.asarray(update.mean[:, 0]), float(update.covariance[0, 0, 0]) + payoff_noise

    full_mean, full_variance = predictive(noise)
    diagonal_mean, diagonal_variance = predictive(np.diag(np.diag(noise)))
    return {
        "draws": draws,
        "seed": seed,
        "noise_covariance": noise.tolist(),
        "coverage": {
            "no_packet": _gaussian_coverage(payoff, prior_mean, prior_variance + payoff_noise),
            "full_R": _gaussian_coverage(payoff, full_mean, full_variance),
            "diagonal_R": _gaussian_coverage(payoff, diagonal_mean, diagonal_variance),
        },
        "predictive_variance": {"full_R": full_variance, "diagonal_R": diagonal_variance},
    }


def unreliable_source(draws=80_000, seed=1, informative_probability=0.6):
    """Informative/null source mixture; reliability model versus always trusting."""

    rng = np.random.default_rng(seed)
    prior_variance, payoff_noise, channel_noise, null_variance = 1.0, 0.2, 0.1, 4.0
    state = rng.normal(0.0, np.sqrt(prior_variance), draws)
    payoff = state + rng.normal(0.0, np.sqrt(payoff_noise), draws)
    informative = rng.random(draws) < informative_probability
    signals = np.where(
        informative,
        state + rng.normal(0.0, np.sqrt(channel_noise), draws),
        rng.normal(0.0, np.sqrt(null_variance), draws),
    )

    m, P, M, R = jnp.zeros(1), jnp.eye(1) * prior_variance, jnp.ones((1, 1)), jnp.eye(1) * channel_noise
    mixture = jax.vmap(
        lambda s: reliability_update(m, P, M, s, R, informative_probability, jnp.zeros(1), jnp.eye(1) * null_variance)
    )(jnp.asarray(signals)[:, None])

    # The payoff predictive is a two-component Gaussian mixture, not a Gaussian.
    weight = np.asarray(mixture.probability)
    weights = np.stack((weight, 1 - weight), axis=1)
    means = np.stack((np.asarray(mixture.informative.mean[:, 0]), np.zeros(draws)), axis=1)
    informative_variance = float(mixture.informative.covariance[0, 0, 0]) + payoff_noise
    standard_deviations = np.sqrt(
        np.stack((np.full(draws, informative_variance), np.full(draws, prior_variance + payoff_noise)), axis=1)
    )
    tail = (1 - COVERAGE) / 2
    lower = _mixture_bounds(weights, means, standard_deviations, tail)
    upper = _mixture_bounds(weights, means, standard_deviations, 1 - tail)
    mixture_coverage = float(np.mean((payoff >= lower) & (payoff <= upper)))
    trusted_coverage = _gaussian_coverage(payoff, means[:, 0], informative_variance)
    return {
        "draws": draws,
        "seed": seed,
        "informative_probability": informative_probability,
        "coverage": {"reliability_mixture": mixture_coverage, "always_trusted": trusted_coverage},
    }


def _plot(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    panels = (
        ("Correlated news: assumed independence", results["correlated_packets"]["coverage"],
         {"no_packet": "No packet", "full_R": "Full R", "diagonal_R": "Diagonal R"}),
        ("Unreliable news: model misspecification", results["unreliable_source"]["coverage"],
         {"reliability_mixture": "Mixture", "always_trusted": "Always trusted"}),
    )
    for axis, (title, coverage, labels) in zip(axes, panels):
        axis.bar([labels[key] for key in coverage], list(coverage.values()), color="#4c72b0")
        axis.axhline(COVERAGE, color="black", linestyle="--", linewidth=1)
        axis.set_ylim(0.4, 1.0)
        axis.set_ylabel("95% interval coverage")
        axis.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    results = {
        "nominal_coverage": COVERAGE,
        "correlated_packets": correlated_packets(),
        "unreliable_source": unreliable_source(),
        "versions": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    _plot(results, RESULTS_DIR / "coverage.png")
    print(json.dumps({key: results[key]["coverage"] for key in ("correlated_packets", "unreliable_source")}, indent=2))


if __name__ == "__main__":
    main()
