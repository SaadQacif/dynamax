"""Synthetic misspecification studies for Section 6 of the research draft.

Run from the repository root::

    python -m fininfo.experiments

Both studies draw from a known generating model and measure the empirical
coverage of nominal 95% predictive intervals for a scalar payoff
``Y = Z + nu``. They test the software against its own assumptions; they say
nothing about real-market predictability. Results, seeds, sample counts and
package versions are written to ``fininfo/results/results.json``.

Each coverage is a mean of independent Bernoulli hits, so it is reported with
its Monte Carlo standard error and a Wilson 95% interval. A correctly specified
predictive law has exact coverage 0.95, so 0.95 should lie inside the interval
for the correct model in about 95% of seeds; the misspecified models are
rejected by many standard errors.
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

from fininfo.algorithms import mixture_quantile  # noqa: E402
from fininfo.core import condition, reliability_payoff, reliability_update  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
COVERAGE = 0.95
QUANTILE_CHECK_DRAWS = 500


def _summary(hits):
    """Empirical coverage with its Monte Carlo standard error and Wilson 95% interval."""

    hits = np.asarray(hits, dtype=bool)
    n, p = hits.size, float(hits.mean())
    z = ndtri(0.975)
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return {
        "coverage": p,
        "standard_error": float(np.sqrt(p * (1 - p) / n)),
        "wilson_95": [float(centre - half), float(centre + half)],
        "z_vs_nominal": float((p - COVERAGE) / np.sqrt(COVERAGE * (1 - COVERAGE) / n)),
    }


def _gaussian_hits(target, mean, variance):
    radius = ndtri((1 + COVERAGE) / 2) * np.sqrt(variance)
    return np.abs(target - mean) <= radius


def _mixture_bounds(weights, means, standard_deviations, probability, iterations=200):
    """Vectorized bisection for the quantile of a per-draw Gaussian mixture.

    The quantile lies between the smallest and largest component quantiles, so
    the bracket below always contains it; 200 halvings reach machine precision.
    """

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
    summaries = {
        "no_packet": _summary(_gaussian_hits(payoff, prior_mean, prior_variance + payoff_noise)),
        "full_R": _summary(_gaussian_hits(payoff, full_mean, full_variance)),
        "diagonal_R": _summary(_gaussian_hits(payoff, diagonal_mean, diagonal_variance)),
    }
    return {
        "draws": draws,
        "seed": seed,
        "parameters": {"prior_variance": prior_variance, "payoff_noise": payoff_noise},
        "noise_covariance": noise.tolist(),
        "coverage": {key: value["coverage"] for key, value in summaries.items()},
        "uncertainty": summaries,
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
    loading, payoff_covariance, offset = jnp.ones((1, 1)), jnp.eye(1) * payoff_noise, jnp.zeros(1)

    def predictive(s):
        reliable = reliability_update(m, P, M, s, R, informative_probability, jnp.zeros(1), jnp.eye(1) * null_variance)
        return reliability_payoff(m, P, reliable, loading, payoff_covariance, offset)

    # The payoff predictive is a two-component Gaussian mixture, not a Gaussian.
    mixture = jax.vmap(predictive)(jnp.asarray(signals)[:, None])
    weights = np.asarray(mixture.weights)
    means = np.asarray(mixture.means[:, :, 0])
    standard_deviations = np.sqrt(np.asarray(mixture.covariances[:, :, 0, 0]))
    tail = (1 - COVERAGE) / 2
    lower = _mixture_bounds(weights, means, standard_deviations, tail)
    upper = _mixture_bounds(weights, means, standard_deviations, 1 - tail)

    # Independent check of the vectorized bisection against Brent's method.
    check = range(QUANTILE_CHECK_DRAWS)
    exact = np.array(
        [
            [mixture_quantile(weights[i], means[i], standard_deviations[i], alpha) for alpha in (tail, 1 - tail)]
            for i in check
        ]
    )
    quantile_error = float(np.abs(exact - np.stack((lower[: len(check)], upper[: len(check)]), axis=1)).max())

    summaries = {
        "reliability_mixture": _summary((payoff >= lower) & (payoff <= upper)),
        "always_trusted": _summary(_gaussian_hits(payoff, means[:, 0], standard_deviations[:, 0] ** 2)),
        # Moment-matched Gaussian: right mean and variance, wrong shape.
        "moment_matched_gaussian": _summary(
            _gaussian_hits(payoff, np.asarray(mixture.mean[:, 0]), np.asarray(mixture.covariance[:, 0, 0]))
        ),
    }
    return {
        "draws": draws,
        "seed": seed,
        "informative_probability": informative_probability,
        "parameters": {
            "prior_variance": prior_variance,
            "payoff_noise": payoff_noise,
            "channel_noise": channel_noise,
            "null_variance": null_variance,
        },
        "coverage": {key: value["coverage"] for key, value in summaries.items()},
        "uncertainty": summaries,
        "quantile_check": {"draws": len(check), "max_abs_error_vs_brentq": quantile_error},
    }


def _plot(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    panels = (
        ("Correlated news: assumed independence", results["correlated_packets"]["uncertainty"],
         {"no_packet": "No packet", "full_R": "Full R", "diagonal_R": "Diagonal R"}),
        ("Unreliable news: model misspecification", results["unreliable_source"]["uncertainty"],
         {"reliability_mixture": "Mixture", "always_trusted": "Always trusted",
          "moment_matched_gaussian": "Moment-matched"}),
    )
    for axis, (title, summaries, labels) in zip(axes, panels):
        values = np.array([summaries[key]["coverage"] for key in labels])
        intervals = np.array([summaries[key]["wilson_95"] for key in labels])
        errors = np.abs(intervals.T - values)
        axis.bar(list(labels.values()), values, yerr=errors, capsize=4, color="#4c72b0")
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
    print(json.dumps({key: results[key]["uncertainty"] for key in ("correlated_packets", "unreliable_source")}, indent=2))
    print("quantile check:", results["unreliable_source"]["quantile_check"])


if __name__ == "__main__":
    main()
