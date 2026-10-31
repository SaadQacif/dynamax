"""Randomized checks of the kernels against independent oracles.

Each test re-derives a result by a route that does not share code with the
kernel under test: Schur complements of the joint covariance, SciPy
densities, numerical quadrature, or direct simulation of the dynamics.
"""

import jax
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import multivariate_normal, t as student_t

from fininfo.core import (
    canonical_relevance,
    cashflow_projection,
    condition,
    edge_attribution,
    linear_payoff,
    normal_interval,
    payoff_information,
    quadratic_decision_value,
    reliability_mixture,
    reliability_payoff,
    reliability_update,
    residualize_second,
    validate_channel,
    validate_covariance,
    validate_payoff,
    whitened_transmission,
)

SEEDS = range(12)


def _spd(rng, n, rank=None):
    rank = n if rank is None else rank
    factor = rng.normal(size=(n, rank))
    return factor @ factor.T + (1e-1 * np.eye(n) if rank == n else 0.0)


def _channel(rng, n=3, r=2, prior_rank=None):
    m = rng.normal(size=n)
    prior = _spd(rng, n, prior_rank)
    measurement = rng.normal(size=(r, n))
    noise = _spd(rng, r)
    signal = rng.normal(size=r)
    return m, prior, measurement, signal, noise


@pytest.mark.parametrize("seed", SEEDS)
def test_condition_matches_joint_gaussian_schur_complement(seed):
    rng = np.random.default_rng(seed)
    m, prior, measurement, signal, noise = _channel(rng)
    validate_channel(m, prior, measurement, signal, noise)
    update = condition(m, prior, measurement, signal, noise)

    # Oracle: condition the joint law of (Z, S) directly.
    cross = prior @ measurement.T
    signal_covariance = measurement @ prior @ measurement.T + noise
    mean_oracle = m + cross @ np.linalg.solve(signal_covariance, signal - measurement @ m)
    covariance_oracle = prior - cross @ np.linalg.solve(signal_covariance, cross.T)
    np.testing.assert_allclose(update.mean, mean_oracle, atol=1e-10)
    np.testing.assert_allclose(update.covariance, covariance_oracle, atol=1e-10)

    # I(Z;S) = h(Z) + h(S) - h(Z,S) when P is nonsingular.
    joint = np.block([[prior, cross], [cross.T, signal_covariance]])
    information_oracle = 0.5 * (
        np.linalg.slogdet(prior)[1] + np.linalg.slogdet(signal_covariance)[1] - np.linalg.slogdet(joint)[1]
    )
    np.testing.assert_allclose(update.information_nats, information_oracle, atol=1e-10)

    evidence_oracle = multivariate_normal(measurement @ m, signal_covariance).logpdf(signal)
    np.testing.assert_allclose(update.log_evidence, evidence_oracle, atol=1e-10)


@pytest.mark.parametrize("seed", SEEDS)
def test_singular_prior_decomposition_and_gain_identity(seed):
    rng = np.random.default_rng(seed)
    m, prior, measurement, signal, noise = _channel(rng, n=4, r=2, prior_rank=2)
    validate_channel(m, prior, measurement, signal, noise)
    update = condition(m, prior, measurement, signal, noise)
    # P = P_+ + D, D = K Sigma K.T, and P_+ is PSD despite the singular prior.
    np.testing.assert_allclose(update.covariance + update.reduction, prior, atol=1e-10)
    np.testing.assert_allclose(
        update.reduction, update.gain @ update.innovation_covariance @ update.gain.T, atol=1e-10
    )
    assert np.linalg.eigvalsh(np.asarray(update.covariance)).min() > -1e-10
    # The posterior mean moves only within the range of the prior covariance.
    null_space = np.linalg.svd(prior)[0][:, 2:]
    np.testing.assert_allclose(null_space.T @ (np.asarray(update.mean) - m), 0.0, atol=1e-10)


@pytest.mark.parametrize("seed", SEEDS)
def test_information_invariances_bounds_and_data_processing(seed):
    rng = np.random.default_rng(seed)
    m, prior, measurement, signal, noise = _channel(rng)
    loading = rng.normal(size=(2, 3))
    payoff_noise = _spd(rng, 2)
    validate_payoff(3, loading, payoff_noise, np.zeros(2))
    update = condition(m, prior, measurement, signal, noise)
    payoff = linear_payoff(m, prior, update, loading, payoff_noise, np.zeros(2))

    relevance = canonical_relevance(payoff.prior_covariance, payoff.revision_covariance)
    information = payoff_information(payoff.prior_covariance, payoff.posterior_covariance)
    np.testing.assert_allclose(relevance.information_nats, information, rtol=1e-9, atol=1e-12)
    assert float(information) <= float(update.information_nats) + 1e-10
    assert float(relevance.lower_bound) <= float(relevance.trace) + 1e-12
    assert float(relevance.trace) <= float(relevance.upper_bound) + 1e-12

    # Invertible state and signal changes leave every financial output fixed.
    state_change = rng.normal(size=(3, 3)) + 3 * np.eye(3)
    signal_change = rng.normal(size=(2, 2)) + 3 * np.eye(2)
    inverse = np.linalg.inv(state_change)
    moved = condition(
        state_change @ m,
        state_change @ prior @ state_change.T,
        signal_change @ measurement @ inverse,
        signal_change @ signal,
        signal_change @ noise @ signal_change.T,
    )
    moved_payoff = linear_payoff(
        state_change @ m, state_change @ prior @ state_change.T, moved, loading @ inverse, payoff_noise, np.zeros(2)
    )
    np.testing.assert_allclose(moved_payoff.revision, payoff.revision, atol=1e-9)
    np.testing.assert_allclose(moved_payoff.revision_covariance, payoff.revision_covariance, atol=1e-9)
    np.testing.assert_allclose(moved.information_nats, update.information_nats, atol=1e-9)
    np.testing.assert_allclose(
        np.linalg.svd(np.asarray(whitened_transmission(moved, loading @ inverse)), compute_uv=False),
        np.linalg.svd(np.asarray(whitened_transmission(update, loading)), compute_uv=False),
        atol=1e-9,
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_residualized_blocks_equal_batch_update(seed):
    rng = np.random.default_rng(seed)
    m, prior, _, _, _ = _channel(rng)
    measurement = rng.normal(size=(4, 3))
    noise = _spd(rng, 4)
    signal = rng.normal(size=4)
    batch = condition(m, prior, measurement, signal, noise)
    first = condition(m, prior, measurement[:2], signal[:2], noise[:2, :2])
    rm2, rs2, rr2 = residualize_second(
        measurement[:2], measurement[2:], signal[:2], signal[2:], noise[:2, :2], noise[2:, :2], noise[2:, 2:]
    )
    sequential = condition(first.mean, first.covariance, rm2, rs2, rr2)
    np.testing.assert_allclose(sequential.mean, batch.mean, atol=1e-9)
    np.testing.assert_allclose(sequential.covariance, batch.covariance, atol=1e-9)
    np.testing.assert_allclose(first.information_nats + sequential.information_nats, batch.information_nats, atol=1e-9)

    naive = condition(first.mean, first.covariance, measurement[2:], signal[2:], noise[2:, 2:])
    assert not np.allclose(naive.mean, batch.mean, atol=1e-6)


@pytest.mark.parametrize("horizon", range(1, 9))
def test_edge_attribution_matches_high_order_quadrature(horizon):
    rng = np.random.default_rng(100 + horizon)
    transition = rng.normal(size=(3, 3)) / 2
    baseline = rng.normal(size=(3, 3)) / 4
    revision, exposure = rng.normal(size=3), rng.normal(size=3)
    attribution = np.asarray(edge_attribution(transition, baseline, revision, exposure, np.int64(horizon)))

    def response(matrix):
        return exposure @ np.linalg.matrix_power(matrix, horizon) @ revision

    np.testing.assert_allclose(attribution.sum(), response(transition) - response(baseline), atol=1e-11)
    # Oracle: 40-node quadrature of the analytic gradient sum_k (F^k).T c d.T (F^{h-1-k}).T.
    nodes, weights = np.polynomial.legendre.leggauss(40)
    integral = np.zeros((3, 3))
    for node, weight in zip((nodes + 1) / 2, weights / 2):
        point = baseline + node * (transition - baseline)
        gradient = sum(
            np.linalg.matrix_power(point, k).T
            @ np.outer(exposure, revision)
            @ np.linalg.matrix_power(point, horizon - 1 - k).T
            for k in range(horizon)
        )
        integral += weight * gradient
    np.testing.assert_allclose(attribution, (transition - baseline) * integral, atol=1e-11)


def test_cashflow_projection_matches_monte_carlo_simulation():
    rng = np.random.default_rng(5)
    transitions = np.array([[[0.9, 0.1], [0.0, 0.8]], [[0.7, 0.2], [-0.1, 0.9]], [[1.0, 0.0], [0.3, 0.6]]])
    process = np.array([_spd(rng, 2) * 0.1 for _ in range(3)])
    biases = rng.normal(size=(3, 2)) * 0.1
    weights = rng.normal(size=(4, 2))
    discounts = 0.97 ** np.arange(4)
    loading, offset, variance = map(np.asarray, cashflow_projection(transitions, process, biases, weights, discounts))

    start = np.array([0.4, -0.3])
    draws = 400_000
    state = np.broadcast_to(start, (draws, 2))
    total = discounts[0] * state @ weights[0]
    for j in range(3):
        noise = rng.multivariate_normal(np.zeros(2), process[j], draws)
        state = state @ transitions[j].T + biases[j] + noise
        total = total + discounts[j + 1] * state @ weights[j + 1]
    np.testing.assert_allclose(total.mean(), loading @ start + offset, atol=4e-3)
    np.testing.assert_allclose(total.var(), variance, rtol=1e-2)


def test_reliability_mixture_with_student_t_null_matches_quadrature():
    prior_mean, prior_variance, channel_variance, probability = 0.3, 1.2, 0.25, 0.4
    null = student_t(df=3, loc=0.0, scale=2.0)
    for signal in (-2.5, 0.1, 3.0):
        update = condition(
            np.array([prior_mean]), np.array([[prior_variance]]), np.eye(1), np.array([signal]),
            np.array([[channel_variance]]),
        )
        mixed = reliability_mixture(
            np.array([prior_mean]), np.array([[prior_variance]]), update, probability, null.logpdf(signal)
        )

        # Oracle: integrate p(z) [pi N(s; z, c) + (1 - pi) f0(s)] over z.
        def joint(z):
            prior_density = np.exp(-0.5 * (z - prior_mean) ** 2 / prior_variance) / np.sqrt(2 * np.pi * prior_variance)
            channel = np.exp(-0.5 * (signal - z) ** 2 / channel_variance) / np.sqrt(2 * np.pi * channel_variance)
            return prior_density * (probability * channel + (1 - probability) * null.pdf(signal))

        evidence = quad(joint, -40, 40, limit=200)[0]
        mean = quad(lambda z: z * joint(z), -40, 40, limit=200)[0] / evidence
        second = quad(lambda z: z * z * joint(z), -40, 40, limit=200)[0] / evidence
        np.testing.assert_allclose(mixed.mean[0], mean, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(mixed.covariance[0, 0], second - mean**2, rtol=1e-8, atol=1e-10)


def test_reliability_degenerate_probabilities_and_payoff_mixture():
    args = (np.zeros(2), np.eye(2), np.array([[1.0, 0.5]]), np.array([4.0]), np.array([[0.3]]))
    trusted = reliability_update(*args, 1.0, np.zeros(1), np.eye(1) * 9.0)
    ignored = reliability_update(*args, 0.0, np.zeros(1), np.eye(1) * 9.0)
    exact = condition(*args)
    np.testing.assert_allclose(trusted.mean, exact.mean, atol=1e-12)
    np.testing.assert_allclose(trusted.covariance, exact.covariance, atol=1e-12)
    np.testing.assert_allclose(ignored.mean, args[0], atol=1e-12)
    np.testing.assert_allclose(ignored.covariance, args[1], atol=1e-12)
    with pytest.raises(ValueError):
        reliability_update(*args, 1.5, np.zeros(1), np.eye(1))

    # A surprising report from a doubtful source can raise posterior variance.
    doubtful = reliability_update(*args, 0.5, np.zeros(1), np.eye(1) * 9.0)
    loading, payoff_noise = np.array([[1.0, -1.0]]), np.array([[0.1]])
    mixture = reliability_payoff(args[0], args[1], doubtful, loading, payoff_noise, np.zeros(1))
    w = np.asarray(mixture.weights)
    means, covariances = np.asarray(mixture.means), np.asarray(mixture.covariances)
    total_mean = w @ means
    total_covariance = sum(w[i] * (covariances[i] + np.outer(means[i] - total_mean, means[i] - total_mean)) for i in range(2))
    np.testing.assert_allclose(mixture.mean, total_mean, atol=1e-12)
    np.testing.assert_allclose(mixture.covariance, total_covariance, atol=1e-12)


def test_kernels_are_jit_compatible_and_reject_bad_concrete_inputs():
    rng = np.random.default_rng(3)
    m, prior, measurement, signal, noise = _channel(rng)
    eager = condition(m, prior, measurement, signal, noise)
    compiled = jax.jit(condition)(m, prior, measurement, signal, noise)
    for left, right in zip(eager, compiled):
        np.testing.assert_allclose(left, right, atol=1e-12)
    np.testing.assert_allclose(jax.jit(normal_interval)(0.0, 1.0, 0.9), normal_interval(0.0, 1.0, 0.9))

    # A non-SPD cost yields NaN, never a finite but meaningless value.
    assert np.isnan(quadratic_decision_value(np.eye(2), np.diag([1.0, -1.0])))
    with pytest.raises(ValueError):
        validate_covariance(np.array([[1.0, 2.0], [2.0, 1.0]]), definite=True)
    with pytest.raises(ValueError):
        validate_payoff(3, np.ones((2, 2)), np.eye(2), np.zeros(2))
    with pytest.raises(ValueError):
        edge_attribution(np.eye(2), np.zeros((2, 2)), np.ones(2), np.ones(2), True)
