"""Tests for attribution, mixture risk, projection, and calibration utilities."""

import numpy as np
import pytest
from scipy.special import ndtr

from fininfo.algorithms import (
    coalition_values,
    conformal_radius,
    discounted_gramian,
    entropy_project,
    mixture_es,
    mixture_quantile,
    price_sensitivity,
    shapley,
)


def test_complementarity_counterexample_and_shapley_efficiency():
    prior = np.eye(2)
    measurement = np.array([[1.0, 1.0], [0.0, 1.0]])
    noise = np.eye(2) * 0.1
    weight = np.diag([1.0, 0.0])
    values = coalition_values(np.zeros(2), prior, measurement, noise, weight)
    np.testing.assert_allclose(values[frozenset({0})], 1 / 2.1)
    np.testing.assert_allclose(values[frozenset({1})], 0.0)
    np.testing.assert_allclose(values[frozenset({0, 1})], 1.1 / 1.31)
    assert values[frozenset({0, 1})] - values[frozenset({0})] > values[frozenset({1})]

    allocation = shapley(values, 2)
    assert (allocation >= 0).all()
    np.testing.assert_allclose(allocation.sum(), values[frozenset({0, 1})])


def test_discounted_gramian_satisfies_lyapunov_equation():
    transition = np.array([[0.8, 0.2], [0.0, 0.7]])
    exposure = np.array([[1.0, -0.3]])
    weight = np.array([[2.0]])
    discount = 0.95
    gramian = discounted_gramian(transition, exposure, weight, discount)
    forcing = exposure.T @ weight @ exposure
    np.testing.assert_allclose(
        gramian, forcing + discount * transition.T @ gramian @ transition, atol=2e-12
    )
    with pytest.raises(ValueError):
        discounted_gramian(np.eye(2) * 1.1, exposure, weight, 1.0)


def test_mixture_quantile_and_es_use_common_var():
    weights = np.array([0.3, 0.7])
    means = np.array([-0.5, 1.2])
    standard_deviations = np.array([0.7, 1.4])
    alpha = 0.95
    quantile = mixture_quantile(weights, means, standard_deviations, alpha)
    cdf = weights @ ndtr((quantile - means) / standard_deviations)
    np.testing.assert_allclose(cdf, alpha, atol=2e-12)
    expected_shortfall = mixture_es(weights, means, standard_deviations, alpha)
    assert expected_shortfall > quantile


def test_entropy_projection_satisfies_constraints_and_kl_identity():
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    features = np.array([[-1.0, 0.2], [0.0, 1.0], [1.0, -0.5], [2.0, 0.7]])
    target = np.array([0.8, 0.25])
    result = entropy_project(prior, features, target)
    weights = result["weights"]
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(weights @ features, target, atol=1e-10)
    assert (weights > 0).all()
    assert result["kl"] >= 0

    with pytest.raises(ValueError):
        entropy_project(prior, features, np.array([5.0, 5.0]))


def test_price_sensitivity_matches_reprojected_finite_difference():
    prior = np.array([0.12, 0.18, 0.25, 0.2, 0.25])
    features = np.array([[-1.0], [-0.2], [0.5], [1.0], [2.0]])
    target = np.array([0.4])
    payoff = np.array([0.2, -0.1, 0.4, 1.3, 2.5])
    result = entropy_project(prior, features, target)
    analytic = price_sensitivity(result["weights"], features, payoff)
    epsilon = 1e-5
    plus = entropy_project(prior, features, target + epsilon)["weights"] @ payoff
    minus = entropy_project(prior, features, target - epsilon)["weights"] @ payoff
    numeric = (plus - minus) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-6)


def test_conformal_radius_uses_finite_sample_index_and_infinity_case():
    scores = np.array([0.2, 0.1, 0.8, 0.4, 0.3])
    np.testing.assert_allclose(conformal_radius(scores, 0.4), 0.4)
    assert np.isinf(conformal_radius(np.array([0.1, 0.2]), 0.05))
    with pytest.raises(ValueError):
        conformal_radius(np.array([-0.1, 0.2]), 0.1)
