"""Numerical checks of the theorem-level diagnostics in Sections 2-4."""

import jax
import numpy as np
import pytest

from fininfo.core import (
    canonical_relevance,
    cara_positions,
    condition,
    linear_payoff,
    mean_error_regret_bound,
    noise_sensitivity,
    payoff_information,
    quadratic_decision_value,
    structural_response,
    weighted_value,
    whitened_transmission,
)


def _example():
    m = np.array([0.2, -0.3, 0.1])
    prior = np.array([[1.1, 0.2, 0.1], [0.2, 0.8, -0.1], [0.1, -0.1, 0.6]])
    measurement = np.array([[1.0, 0.4, 0.0], [0.0, 0.5, 1.0]])
    signal = np.array([0.9, -0.4])
    noise = np.array([[0.5, 0.15], [0.15, 0.7]])
    loading = np.array([[1.0, -0.5, 0.3], [0.2, 0.9, -0.4]])
    payoff_noise = np.array([[0.3, 0.05], [0.05, 0.2]])
    update = condition(m, prior, measurement, signal, noise)
    payoff = linear_payoff(m, prior, update, loading, payoff_noise, np.zeros(2))
    return m, prior, measurement, signal, noise, loading, update, payoff


def test_canonical_relevance_matches_information_and_bounds():
    *_, update, payoff = _example()
    relevance = canonical_relevance(payoff.prior_covariance, payoff.revision_covariance)
    information = payoff_information(payoff.prior_covariance, payoff.posterior_covariance)
    np.testing.assert_allclose(relevance.information_nats, information, rtol=1e-12)
    assert (relevance.squared_correlations >= -1e-14).all()
    assert (relevance.squared_correlations < 1).all()
    assert relevance.lower_bound <= relevance.trace + 1e-14
    assert relevance.trace <= relevance.upper_bound + 1e-14
    # Data processing: payoff information cannot exceed state information.
    assert float(information) <= float(update.information_nats) + 1e-12


def test_canonical_relevance_scalar_bounds_coincide():
    relevance = canonical_relevance(np.array([[2.0]]), np.array([[0.5]]))
    np.testing.assert_allclose(relevance.trace, 0.25)
    np.testing.assert_allclose(relevance.lower_bound, relevance.upper_bound)
    np.testing.assert_allclose(relevance.lower_bound, relevance.trace)


def test_whitened_transmission_reproduces_revision_and_is_root_invariant():
    m, prior, measurement, signal, noise, loading, update, payoff = _example()
    transmission = whitened_transmission(update, loading)
    np.testing.assert_allclose(transmission @ transmission.T, payoff.revision_covariance, atol=2e-12)
    chol = np.linalg.cholesky(np.asarray(update.innovation_covariance))
    whitened_innovation = np.linalg.solve(chol, np.asarray(update.innovation))
    np.testing.assert_allclose(transmission @ whitened_innovation, payoff.revision, atol=2e-12)

    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    other_root = np.asarray(loading @ update.gain) @ chol @ rotation
    np.testing.assert_allclose(
        np.linalg.svd(other_root, compute_uv=False),
        np.linalg.svd(np.asarray(transmission), compute_uv=False),
        atol=2e-12,
    )


def test_noise_sensitivity_matches_finite_difference_and_is_nonpositive():
    _, prior, measurement, _, noise, *_ = _example()
    weight = np.array([[1.0, 0.1, 0.0], [0.1, 0.5, 0.2], [0.0, 0.2, 0.9]])
    perturbation = np.array([[0.3, 0.1], [0.1, 0.4]])

    def value(current_noise):
        update = condition(np.zeros(3), prior, measurement, np.zeros(2), current_noise)
        return float(weighted_value(update.reduction, weight))

    epsilon = 1e-5
    numeric = (value(noise + epsilon * perturbation) - value(noise - epsilon * perturbation)) / (2 * epsilon)
    analytic = float(noise_sensitivity(prior, measurement, noise, weight, perturbation))
    np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-9)
    assert analytic <= 0


def test_cara_indifference_identity_by_monte_carlo():
    rng = np.random.default_rng(7)
    mean = np.array([0.3, -0.2])
    prior = np.array([[1.0, 0.3], [0.3, 0.8]])
    posterior = np.array([[0.6, 0.1], [0.1, 0.5]])
    risk_aversion = 2.0

    # Optimized CARA utility is -exp(-gamma w - mu.T V^{-1} mu / 2); w = 0 here.
    def optimized_exponent(mu, covariance):
        position = np.asarray(cara_positions(mu, covariance, risk_aversion))
        return -risk_aversion * position @ mu + 0.5 * risk_aversion**2 * position @ covariance @ position

    no_signal = np.exp(optimized_exponent(mean, prior))
    conditional_means = rng.multivariate_normal(mean, prior - posterior, size=200_000)
    precision = np.linalg.inv(posterior)
    with_signal = np.exp(-0.5 * np.einsum("ni,ij,nj->n", conditional_means, precision, conditional_means)).mean()
    expected_ratio = np.sqrt(np.linalg.det(posterior) / np.linalg.det(prior))
    np.testing.assert_allclose(with_signal / no_signal, expected_ratio, rtol=5e-3)
    np.testing.assert_allclose(optimized_exponent(mean, prior), -0.5 * mean @ np.linalg.solve(prior, mean))


def _box_value(mean, diagonal_cost, lower, upper):
    position = np.clip(mean / diagonal_cost, lower, upper)
    return (position * mean - 0.5 * diagonal_cost * position**2).sum(axis=-1)


def test_constrained_decision_value_bounds_and_regret():
    rng = np.random.default_rng(11)
    diagonal_cost = np.array([2.0, 1.5])
    cost = np.diag(diagonal_cost)
    mean = np.array([0.4, -0.1])
    revision_covariance = np.array([[0.5, 0.1], [0.1, 0.3]])
    draws = rng.multivariate_normal(mean, revision_covariance, size=200_000)
    bound = float(quadratic_decision_value(revision_covariance, cost))

    constrained_gain = _box_value(draws, diagonal_cost, -0.2, 0.3).mean() - _box_value(mean, diagonal_cost, -0.2, 0.3)
    unrestricted_gain = _box_value(draws, diagonal_cost, -np.inf, np.inf).mean() - _box_value(
        mean, diagonal_cost, -np.inf, np.inf
    )
    assert 0 <= constrained_gain <= bound
    np.testing.assert_allclose(unrestricted_gain, bound, rtol=1e-2)

    used_mean = np.array([0.1, 0.25])
    position = np.linalg.solve(cost, used_mean)
    achieved = position @ mean - 0.5 * position @ cost @ position
    optimum = 0.5 * mean @ np.linalg.solve(cost, mean)
    np.testing.assert_allclose(optimum - achieved, mean_error_regret_bound(cost, mean, used_mean), rtol=1e-12)


def test_structural_response_matches_simulated_intervention():
    transition = np.array([[0.9, 0.2], [-0.1, 0.8]])
    control = np.array([[1.0], [0.5]])
    control_input = np.array([0.3])
    horizon = 4
    state_with, state_without = np.zeros(2), np.zeros(2)
    for step in range(horizon):
        shock = control @ control_input if step == 0 else 0.0
        state_with = transition @ state_with + shock
        state_without = transition @ state_without
    np.testing.assert_allclose(
        structural_response(transition, control, control_input, horizon), state_with - state_without, atol=1e-14
    )
    with pytest.raises(ValueError):
        structural_response(transition, control, control_input, 0)


def test_new_kernels_compile_under_jit():
    *_, update, payoff = _example()
    jitted = jax.jit(canonical_relevance)(payoff.prior_covariance, payoff.revision_covariance)
    np.testing.assert_allclose(jitted.trace, canonical_relevance(payoff.prior_covariance, payoff.revision_covariance).trace)
