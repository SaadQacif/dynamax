"""Tests for the exact Gaussian and financial decision kernels."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fininfo.core import (
    cara_information_fee,
    cashflow_projection,
    condition,
    edge_attribution,
    ensemble_weights,
    from_filtered,
    linear_payoff,
    normal_es,
    normal_interval,
    payoff_information,
    propagate,
    quadratic_decision_value,
    quadratic_moments,
    reliability_update,
    residualize_second,
    validate_channel,
    weighted_value,
)


def test_condition_supports_singular_prior_and_preserves_decomposition():
    m = np.array([0.2, -0.4])
    prior = np.array([[1.0, 1.0], [1.0, 1.0]])
    measurement = np.array([[1.0, -0.2]])
    signal = np.array([0.7])
    noise = np.array([[0.3]])
    validate_channel(m, prior, measurement, signal, noise)

    update = condition(m, prior, measurement, signal, noise)
    np.testing.assert_allclose(update.covariance + update.reduction, prior, atol=2e-12)
    assert np.linalg.eigvalsh(update.covariance).min() >= -2e-12
    assert float(update.information_nats) >= 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("prior", np.array([[1.0, 0.2], [0.1, 1.0]])),
        ("prior", np.array([[1.0, 0.0], [0.0, -0.1]])),
        ("noise", np.array([[0.0]])),
    ],
)
def test_validate_channel_rejects_invalid_covariances(field, value):
    prior = np.eye(2)
    noise = np.eye(1)
    if field == "prior":
        prior = value
    else:
        noise = value
    with pytest.raises(ValueError):
        validate_channel(np.zeros(2), prior, np.ones((1, 2)), np.zeros(1), noise)


def test_condition_is_jittable_and_state_coordinate_invariant():
    m = jnp.array([0.4, -0.1])
    prior = jnp.array([[1.2, 0.3], [0.3, 0.8]])
    measurement = jnp.array([[1.0, -0.3], [0.2, 0.7]])
    signal = jnp.array([0.8, -0.5])
    noise = jnp.array([[0.4, 0.12], [0.12, 0.7]])
    loading = jnp.array([[1.5, -0.4]])
    payoff_noise = jnp.array([[0.2]])
    offset = jnp.array([0.3])

    update = jax.jit(condition)(m, prior, measurement, signal, noise)
    payoff = linear_payoff(m, prior, update, loading, payoff_noise, offset)

    transform = jnp.array([[1.2, 0.4], [-0.3, 0.9]])
    inverse = jnp.linalg.inv(transform)
    transformed = condition(
        transform @ m,
        transform @ prior @ transform.T,
        measurement @ inverse,
        signal,
        noise,
    )
    transformed_payoff = linear_payoff(
        transform @ m,
        transform @ prior @ transform.T,
        transformed,
        loading @ inverse,
        payoff_noise,
        offset,
    )
    np.testing.assert_allclose(transformed_payoff.revision, payoff.revision, atol=2e-12)
    np.testing.assert_allclose(
        transformed_payoff.revision_covariance, payoff.revision_covariance, atol=2e-12
    )


def test_invertible_signal_change_preserves_financial_outputs():
    m = np.array([0.1, 0.2])
    prior = np.array([[1.0, 0.25], [0.25, 0.7]])
    measurement = np.array([[1.0, 0.0], [0.4, 1.0]])
    signal = np.array([0.7, -0.3])
    noise = np.array([[0.5, 0.2], [0.2, 0.8]])
    change = np.array([[1.0, 0.3], [-0.2, 1.1]])

    original = condition(m, prior, measurement, signal, noise)
    changed = condition(m, prior, change @ measurement, change @ signal, change @ noise @ change.T)
    np.testing.assert_allclose(changed.mean, original.mean, atol=2e-12)
    np.testing.assert_allclose(changed.covariance, original.covariance, atol=2e-12)
    np.testing.assert_allclose(changed.information_nats, original.information_nats, atol=2e-12)


def test_information_relevance_and_scalar_identity():
    # The signal observes only z_2 while the payoff loads only z_1.
    m = np.zeros(2)
    prior = np.eye(2)
    update = condition(m, prior, np.array([[0.0, 1.0]]), np.zeros(1), np.array([[0.01]]))
    payoff = linear_payoff(m, prior, update, np.array([[1.0, 0.0]]), np.array([[0.2]]), np.zeros(1))
    assert float(update.information_nats) > 2.0
    np.testing.assert_allclose(payoff.revision_covariance, 0.0, atol=1e-13)
    np.testing.assert_allclose(payoff_information(payoff.prior_covariance, payoff.posterior_covariance), 0.0)

    relevant = linear_payoff(m, prior, update, np.array([[0.0, 1.0]]), np.array([[0.2]]), np.zeros(1))
    information = float(payoff_information(relevant.prior_covariance, relevant.posterior_covariance))
    variance = float(relevant.prior_covariance[0, 0])
    expected_reduction = variance * (1 - np.exp(-2 * information))
    np.testing.assert_allclose(relevant.revision_covariance[0, 0], expected_reduction, rtol=1e-12)


def test_measurement_noise_sensitivity_matches_finite_difference():
    prior = np.array([[1.3, 0.2], [0.2, 0.9]])
    measurement = np.array([[1.0, 0.3], [-0.2, 0.8]])
    noise = np.array([[0.6, 0.1], [0.1, 0.5]])
    perturbation = np.array([[0.4, 0.05], [0.05, 0.2]])
    weight = np.array([[1.0, 0.2], [0.2, 0.7]])

    def value(current_noise):
        update = condition(np.zeros(2), prior, measurement, np.zeros(2), current_noise)
        return float(weighted_value(update.reduction, weight))

    epsilon = 1e-5
    numeric = (value(noise + epsilon * perturbation) - value(noise - epsilon * perturbation)) / (2 * epsilon)
    innovation = measurement @ prior @ measurement.T + noise
    analytic = -np.trace(
        np.linalg.solve(innovation, measurement @ prior @ weight @ prior @ measurement.T)
        @ np.linalg.solve(innovation, perturbation)
    )
    np.testing.assert_allclose(numeric, analytic, rtol=2e-6, atol=2e-8)
    assert analytic <= 0


def test_decision_and_cara_values_have_exact_gaussian_forms():
    reduction = np.array([[0.4, 0.05], [0.05, 0.2]])
    cost = np.array([[2.0, 0.1], [0.1, 1.4]])
    expected = 0.5 * np.trace(np.linalg.solve(cost, reduction))
    np.testing.assert_allclose(quadratic_decision_value(reduction, cost), expected)

    prior = np.array([[1.2, 0.2], [0.2, 0.9]])
    posterior = prior - reduction
    information = float(payoff_information(prior, posterior))
    np.testing.assert_allclose(cara_information_fee(prior, posterior, 3.0), information / 3.0)
    assert information > 0


def test_normal_interval_and_expected_shortfall_edge_cases():
    np.testing.assert_allclose(normal_interval(2.0, 0.0), np.array([2.0, 2.0]))
    np.testing.assert_allclose(normal_es(2.0, 0.0), 2.0)
    with pytest.raises(ValueError):
        normal_interval(0.0, 1.0, coverage=1.0)


def test_propagate_respects_chronological_order_and_horizon_zero():
    m = np.array([1.0, -1.0])
    prior = np.array([[0.5, 0.1], [0.1, 0.8]])
    transitions = np.array([[[1.0, 1.0], [0.0, 1.0]], [[0.8, 0.0], [0.2, 0.9]]])
    process = np.array([np.eye(2) * 0.1, np.eye(2) * 0.2])
    biases = np.array([[0.1, 0.0], [-0.2, 0.3]])
    mean, covariance, propagator, future_noise = propagate(m, prior, transitions, process, biases)
    expected_propagator = transitions[1] @ transitions[0]
    np.testing.assert_allclose(propagator, expected_propagator)
    np.testing.assert_allclose(covariance, expected_propagator @ prior @ expected_propagator.T + future_noise)

    empty = np.empty((0, 2, 2))
    result = propagate(m, prior, empty, empty, np.empty((0, 2)))
    np.testing.assert_allclose(result[0], m)
    np.testing.assert_allclose(result[1], prior)
    np.testing.assert_allclose(result[2], np.eye(2))
    np.testing.assert_allclose(result[3], np.zeros((2, 2)))


def test_residualized_sequential_update_equals_correlated_batch_update():
    m = np.array([0.2, -0.1])
    prior = np.array([[1.0, 0.2], [0.2, 0.7]])
    m1 = np.array([[1.0, 0.0]])
    m2 = np.array([[0.3, 1.0]])
    s1, s2 = np.array([0.8]), np.array([-0.5])
    r11, r22, r21 = np.array([[0.6]]), np.array([[0.9]]), np.array([[0.25]])
    full_noise = np.block([[r11, r21.T], [r21, r22]])
    batch = condition(m, prior, np.vstack((m1, m2)), np.r_[s1, s2], full_noise)

    first = condition(m, prior, m1, s1, r11)
    rm2, rs2, rr2 = residualize_second(m1, m2, s1, s2, r11, r21, r22)
    sequential = condition(first.mean, first.covariance, rm2, rs2, rr2)
    np.testing.assert_allclose(sequential.mean, batch.mean, atol=2e-12)
    np.testing.assert_allclose(sequential.covariance, batch.covariance, atol=2e-12)


def test_cashflow_projection_matches_direct_coefficients():
    transitions = np.array([[[0.8, 0.2], [0.0, 0.9]], [[0.7, -0.1], [0.1, 0.8]]])
    process = np.array([[[0.2, 0.03], [0.03, 0.1]], [[0.15, 0.0], [0.0, 0.12]]])
    biases = np.array([[0.1, -0.2], [0.05, 0.03]])
    weights = np.array([[1.0, 0.0], [0.5, 0.4], [-0.2, 0.7]])
    discounts = np.array([1.0, 0.95, 0.9])
    loading, offset, variance = cashflow_projection(transitions, process, biases, weights, discounts)

    c0, c1, c2 = weights * discounts[:, None]
    b2 = c2
    b1 = c1 + transitions[1].T @ b2
    expected_loading = c0 + transitions[0].T @ b1
    expected_offset = b1 @ biases[0] + b2 @ biases[1]
    expected_variance = b1 @ process[0] @ b1 + b2 @ process[1] @ b2
    np.testing.assert_allclose(loading, expected_loading)
    np.testing.assert_allclose(offset, expected_offset)
    np.testing.assert_allclose(variance, expected_variance)


def test_quadratic_moments_match_monte_carlo():
    rng = np.random.default_rng(22)
    m = np.array([0.3, -0.4])
    prior = np.array([[0.8, 0.2], [0.2, 0.5]])
    linear = np.array([0.7, -0.1])
    quadratic = np.array([[0.4, 0.15], [0.15, -0.2]])
    expected_mean, expected_variance = quadratic_moments(m, prior, linear, quadratic, offset=0.2)
    draws = rng.multivariate_normal(m, prior, size=300_000)
    values = 0.2 + draws @ linear + 0.5 * np.einsum("ni,ij,nj->n", draws, quadratic, draws)
    np.testing.assert_allclose(values.mean(), expected_mean, atol=5e-3)
    np.testing.assert_allclose(values.var(), expected_variance, rtol=1.5e-2)


def test_reliability_update_matches_total_mixture_moments():
    m = np.array([0.0])
    prior = np.array([[1.0]])
    result = reliability_update(
        m,
        prior,
        np.array([[1.0]]),
        np.array([2.0]),
        np.array([[0.2]]),
        0.35,
        np.array([0.0]),
        np.array([[4.0]]),
    )
    weight = float(result.probability)
    informative_mean = float(result.informative.mean[0])
    expected_mean = weight * informative_mean
    expected_variance = (
        weight * float(result.informative.covariance[0, 0])
        + (1 - weight)
        + weight * (1 - weight) * informative_mean**2
    )
    np.testing.assert_allclose(result.mean[0], expected_mean)
    np.testing.assert_allclose(result.covariance[0, 0], expected_variance)


def test_edge_attribution_is_complete_and_basis_dependent():
    transition = np.array([[0.8, 0.4], [-0.2, 0.9]])
    baseline = np.zeros((2, 2))
    revision = np.array([0.7, -0.3])
    exposure = np.array([1.2, 0.5])
    horizon = 4
    attribution = edge_attribution(transition, baseline, revision, exposure, horizon)
    response = exposure @ np.linalg.matrix_power(transition, horizon) @ revision
    baseline_response = exposure @ np.linalg.matrix_power(baseline, horizon) @ revision
    np.testing.assert_allclose(attribution.sum(), response - baseline_response, atol=2e-12)


def test_ensemble_weights_and_filtered_adapter():
    weights = ensemble_weights(np.array([0.2, 0.8]), np.log(np.array([0.6, 0.1])))
    np.testing.assert_allclose(weights, np.array([0.6, 0.4]))
    filtered = SimpleNamespace(
        filtered_means=np.array([[0.0], [0.4]]),
        filtered_covariances=np.array([[[1.0]], [[0.5]]]),
    )
    direct = condition(np.array([0.4]), np.array([[0.5]]), np.array([[1.0]]), np.array([1.0]), np.array([[0.2]]))
    adapted = from_filtered(filtered, 1, np.array([[1.0]]), np.array([1.0]), np.array([[0.2]]))
    np.testing.assert_allclose(adapted.mean, direct.mean)
