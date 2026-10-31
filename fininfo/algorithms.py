"""Eager reference algorithms for attribution, mixtures, and projection.

These NumPy/SciPy routines are deliberately separate from the pure JAX kernels.
They are designed for auditability and small reference problems, not advertised
as JIT-compatible production implementations.
"""

from itertools import combinations
from math import ceil, factorial

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import brentq, linprog
from scipy.special import logsumexp, ndtr, ndtri

from fininfo.core import condition


def coalition_values(m, prior_covariance, measurement, noise_covariance, weight):
    """Compute all coalition values when each scalar signal is one player.

    Exact enumeration is limited to 12 signals. Submatrices of the full noise
    covariance are used, so correlated measurement errors are retained.
    Production attribution should group observations by immutable packet or
    source identity rather than by an arbitrary coordinate split.
    """

    measurement = np.asarray(measurement)
    noise_covariance = np.asarray(noise_covariance)
    weight = np.asarray(weight)
    players = len(measurement)
    if players > 12:
        raise ValueError("exact attribution is limited to 12 scalar signals")
    if measurement.ndim != 2 or noise_covariance.shape != (players, players):
        raise ValueError("incompatible measurement and noise-covariance dimensions")
    if weight.shape != np.asarray(prior_covariance).shape:
        raise ValueError("weight must have the same shape as the prior covariance")
    values = {frozenset(): 0.0}
    for size in range(1, players + 1):
        for ids in combinations(range(players), size):
            index = np.asarray(ids)
            loading = measurement[index]
            update = condition(
                m,
                prior_covariance,
                loading,
                loading @ np.asarray(m),
                noise_covariance[np.ix_(index, index)],
            )
            values[frozenset(ids)] = float(np.trace(weight @ np.asarray(update.reduction)))
    return values


def shapley(values, players: int):
    """Return the exact Shapley allocation of a complete coalition dictionary."""

    if not isinstance(players, int) or players < 1:
        raise ValueError("players must be a positive integer")
    expected = {frozenset(ids) for size in range(players + 1) for ids in combinations(range(players), size)}
    if set(values) != expected:
        missing = expected - set(values)
        extra = set(values) - expected
        raise ValueError(f"coalition dictionary is incomplete (missing={len(missing)}, extra={len(extra)})")
    allocation = np.zeros(players)
    for player in range(players):
        for coalition, value in values.items():
            if player not in coalition:
                size = len(coalition)
                coefficient = factorial(size) * factorial(players - size - 1) / factorial(players)
                allocation[player] += coefficient * (values[coalition | {player}] - value)
    return allocation


def discounted_gramian(transition, exposure, weight, discount):
    """Compute an infinite-horizon discounted state-exposure Gramian.

    The series is
    ``sum_h discount**h (F**h).T C.T weight C F**h``. Stability of
    ``sqrt(discount) * F`` is checked explicitly.
    """

    if not 0 < discount <= 1:
        raise ValueError("discount must lie in (0, 1]")
    transition, exposure, weight = map(np.asarray, (transition, exposure, weight))
    scaled = np.sqrt(discount) * transition
    if max(abs(np.linalg.eigvals(scaled))) >= 1:
        raise ValueError("discounted dynamics are not stable")
    forcing = exposure.T @ weight @ exposure
    return solve_discrete_lyapunov(scaled.T, forcing)


def _mixture_inputs(weights, means, standard_deviations):
    weights, means, standard_deviations = map(
        lambda array: np.asarray(array, dtype=float), (weights, means, standard_deviations)
    )
    if weights.ndim != 1 or means.shape != weights.shape or standard_deviations.shape != weights.shape:
        raise ValueError("mixture inputs must have equal nonempty vector shapes")
    if not len(weights):
        raise ValueError("mixture inputs must be nonempty")
    if not all(np.isfinite(array).all() for array in (weights, means, standard_deviations)):
        raise ValueError("non-finite mixture input")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be nonnegative and sum to one")
    if (standard_deviations <= 0).any():
        raise ValueError("component standard deviations must be positive")
    return weights, means, standard_deviations


def mixture_quantile(weights, means, standard_deviations, alpha):
    """Return an exact one-dimensional Gaussian-mixture quantile."""

    weights, means, standard_deviations = _mixture_inputs(weights, means, standard_deviations)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    component_quantiles = means + standard_deviations * ndtri(alpha)
    lower, upper = component_quantiles.min(), component_quantiles.max()
    if lower == upper:
        return float(lower)
    return brentq(
        lambda value: weights @ ndtr((value - means) / standard_deviations) - alpha,
        lower,
        upper,
        xtol=1e-12,
        rtol=1e-12,
    )


def mixture_es(weights, means, standard_deviations, alpha=0.95):
    """Return upper-tail mixture ES at the common mixture VaR."""

    weights, means, standard_deviations = _mixture_inputs(weights, means, standard_deviations)
    quantile = mixture_quantile(weights, means, standard_deviations, alpha)
    standardized = (quantile - means) / standard_deviations
    tail_first_moment = means * ndtr(-standardized) + standard_deviations * np.exp(
        -(standardized**2) / 2
    ) / np.sqrt(2 * np.pi)
    return float(weights @ tail_first_moment / (1 - alpha))


def entropy_project(prior, features, target, tol=1e-10, max_iterations=100):
    """Project finite scenarios by minimizing ``KL(q || prior)``.

    Constraints are ``q @ features == target`` and ``q.sum() == 1``. The
    implementation requires a strictly positive prior, affine-independent
    feature columns, and a target in the numerical relative interior of the
    finite feature hull. Boundary and infeasible targets are rejected rather
    than silently regularized. This is a probability projection, not an
    automatic construction of a risk-neutral pricing measure.
    """

    prior, features, target = map(lambda array: np.asarray(array, dtype=float), (prior, features, target))
    if prior.ndim != 1 or features.ndim != 2:
        raise ValueError("prior must be a vector and features must be a matrix")
    if features.shape[0] != len(prior) or target.shape != (features.shape[1],):
        raise ValueError("incompatible projection dimensions")
    if not all(np.isfinite(array).all() for array in (prior, features, target)):
        raise ValueError("non-finite projection input")
    if (prior <= 0).any() or not np.isclose(prior.sum(), 1.0):
        raise ValueError("prior must be strictly positive and normalized")
    scenarios, dimensions = features.shape
    if dimensions == 0:
        return {
            "weights": prior.copy(),
            "dual": np.empty(0),
            "residual": np.empty(0),
            "kl": 0.0,
            "iterations": 0,
        }
    if np.linalg.matrix_rank(features[1:] - features[0]) != dimensions:
        raise ValueError("remove affine-redundant feature columns")

    equality = np.vstack((np.ones(scenarios), features.T))
    # Maximize t subject to every scenario probability q_i >= t.
    feasibility = linprog(
        np.r_[np.zeros(scenarios), -1.0],
        A_ub=np.c_[-np.eye(scenarios), np.ones(scenarios)],
        b_ub=np.zeros(scenarios),
        A_eq=np.c_[equality, np.zeros(dimensions + 1)],
        b_eq=np.r_[1.0, target],
        bounds=[(0, None)] * (scenarios + 1),
        method="highs",
    )
    if not feasibility.success or feasibility.x[-1] <= tol:
        raise ValueError("target is not numerically strictly feasible")

    log_prior = np.log(prior)
    dual = np.zeros(dimensions)

    def objective(value):
        return logsumexp(log_prior + features @ value) - target @ value

    for iteration in range(max_iterations):
        log_weights = log_prior + features @ dual
        weights = np.exp(log_weights - logsumexp(log_weights))
        mean = weights @ features
        residual = mean - target
        if np.max(abs(residual)) <= tol:
            return {
                "weights": weights,
                "dual": dual,
                "residual": residual,
                "kl": float(weights @ (np.log(weights) - log_prior)),
                "iterations": iteration,
            }
        centered = features - mean
        covariance = centered.T @ (weights[:, None] * centered)
        try:
            direction = np.linalg.solve(covariance, residual)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("dual Hessian became singular") from exc
        step = 1.0
        old_objective = objective(dual)
        directional_decrement = residual @ direction
        while objective(dual - step * direction) > old_objective - 1e-4 * step * directional_decrement:
            step *= 0.5
            if step < 2**-40:
                raise RuntimeError("dual line search failed")
        dual -= step * direction
    raise RuntimeError("projection failed to meet the moment tolerance")


def price_sensitivity(weights, features, payoff):
    """Differentiate the projected expected payoff with respect to target moments."""

    weights, features, payoff = map(np.asarray, (weights, features, payoff))
    if weights.ndim != 1 or features.ndim != 2 or payoff.shape != weights.shape:
        raise ValueError("incompatible sensitivity dimensions")
    if features.shape[0] != len(weights):
        raise ValueError("one feature row is required per scenario")
    centered_features = features - weights @ features
    feature_covariance = centered_features.T @ (weights[:, None] * centered_features)
    cross_covariance = centered_features.T @ (weights * (payoff - weights @ payoff))
    return np.linalg.solve(feature_covariance, cross_covariance)


def conformal_radius(scores, alpha):
    """Return the split-conformal order statistic under exchangeability.

    If the finite-sample index is ``n + 1``, infinity is returned rather than
    an invalid finite guarantee. Serial financial observations are not
    exchangeable merely because this function is applied to them.
    """

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite nonempty vector")
    if (scores < 0).any() or not 0 < alpha < 1:
        raise ValueError("need nonnegative scores and 0 < alpha < 1")
    index = ceil((len(scores) + 1) * (1 - alpha))
    return np.inf if index > len(scores) else float(np.partition(scores, index - 1)[index - 1])
