"""Integration tests against actual Dynamax filtering outputs."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from fininfo.core import cashflow_projection, condition, from_filtered, linear_payoff


def test_new_packet_update_uses_filtered_not_smoothed_marginal():
    model = LinearGaussianSSM(state_dim=2, emission_dim=1)
    params, _ = model.initialize(
        jr.PRNGKey(1),
        initial_mean=jnp.array([0.0, 0.0]),
        initial_covariance=jnp.array([[1.0, 0.2], [0.2, 0.7]]),
        dynamics_weights=jnp.array([[0.9, 0.1], [0.0, 0.8]]),
        dynamics_covariance=jnp.eye(2) * 0.1,
        emission_weights=jnp.array([[1.0, -0.2]]),
        emission_covariance=jnp.array([[0.4]]),
    )
    emissions = jnp.array([[0.2], [-0.4], [0.8]])
    filtered = model.filter(params, emissions)
    time_index = 1
    packet_loading = jnp.array([[0.3, 1.0]])
    packet = jnp.array([0.6])
    packet_noise = jnp.array([[0.25]])

    integrated = from_filtered(filtered, time_index, packet_loading, packet, packet_noise)
    direct = condition(
        filtered.filtered_means[time_index],
        filtered.filtered_covariances[time_index],
        packet_loading,
        packet,
        packet_noise,
    )
    np.testing.assert_allclose(integrated.mean, direct.mean, atol=2e-12)
    np.testing.assert_allclose(integrated.covariance, direct.covariance, atol=2e-12)

    payoff = linear_payoff(
        filtered.filtered_means[time_index],
        filtered.filtered_covariances[time_index],
        integrated,
        jnp.array([[1.0, 0.5]]),
        jnp.array([[0.15]]),
        jnp.array([0.0]),
    )
    assert float(payoff.revision_covariance[0, 0]) > 0


def test_time_varying_transition_into_t_uses_parameter_entry_t():
    model = LinearGaussianSSM(state_dim=1, emission_dim=1)
    params, _ = model.initialize(
        jr.PRNGKey(2),
        initial_mean=jnp.array([0.0]),
        initial_covariance=jnp.array([[1.0]]),
        dynamics_weights=jnp.array([[[9.0]], [[2.0]]]),
        dynamics_bias=jnp.zeros((2, 1)),
        dynamics_covariance=jnp.array([[[0.1]], [[0.1]]]),
        emission_weights=jnp.ones((2, 1, 1)),
        emission_bias=jnp.zeros((2, 1)),
        emission_covariance=jnp.array([[[0.5]], [[0.5]]]),
    )
    filtered = model.filter(params, jnp.array([[1.0], [0.0]]))

    posterior0_mean = 1.0 / 1.5
    posterior0_variance = 1.0 - 1.0 / 1.5
    predicted1_mean = 2.0 * posterior0_mean
    predicted1_variance = 4.0 * posterior0_variance + 0.1
    gain1 = predicted1_variance / (predicted1_variance + 0.5)
    expected_filtered1_mean = predicted1_mean * (1 - gain1)
    expected_filtered1_variance = predicted1_variance * (1 - gain1)
    np.testing.assert_allclose(filtered.filtered_means[1, 0], expected_filtered1_mean, atol=1e-12)
    np.testing.assert_allclose(filtered.filtered_covariances[1, 0, 0], expected_filtered1_variance, atol=1e-12)


def test_cashflow_projection_retains_cross_horizon_dependence():
    transition = np.array([[[0.8]], [[0.7]]])
    process = np.array([[[0.3]], [[0.2]]])
    biases = np.zeros((2, 1))
    weights = np.ones((3, 1))
    discounts = np.array([1.0, 0.95, 0.9])
    loading, _, future_variance = cashflow_projection(transition, process, biases, weights, discounts)

    # Directly expand X = c0*z0 + c1*z1 + c2*z2. The first process shock
    # enters both z1 and z2, so its loading must be squared after aggregation.
    c0, c1, c2 = discounts
    expected_loading = c0 + c1 * 0.8 + c2 * 0.7 * 0.8
    shock1_loading = c1 + c2 * 0.7
    shock2_loading = c2
    expected_future_variance = shock1_loading**2 * 0.3 + shock2_loading**2 * 0.2
    naive_sum_of_marginals = c1**2 * 0.3 + c2**2 * (0.7**2 * 0.3 + 0.2)
    np.testing.assert_allclose(loading[0], expected_loading)
    np.testing.assert_allclose(future_variance, expected_future_variance)
    assert not np.isclose(float(future_variance), naive_sum_of_marginals)
