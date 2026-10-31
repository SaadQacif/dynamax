"""Pure JAX kernels for decision-aware Gaussian information transport.

All probability distributions are conditional on the same pre-signal
information set. Covariances and likelihood parameters are fixed before the
new signal is observed. The kernels perform no hidden jitter, clipping, or
nearest-positive-definite repair.

Call :func:`validate_channel` before using a compiled update with external
data. The validation routine is intentionally eager and is not JIT compatible.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.scipy.special import logsumexp, ndtri


class Update(NamedTuple):
    """Outputs of an exact linear-Gaussian measurement update."""

    mean: jax.Array
    covariance: jax.Array
    gain: jax.Array
    innovation: jax.Array
    innovation_covariance: jax.Array
    reduction: jax.Array
    information_nats: jax.Array
    log_evidence: jax.Array


class Payoff(NamedTuple):
    """Prior and posterior moments of a linear financial exposure."""

    prior_mean: jax.Array
    posterior_mean: jax.Array
    prior_covariance: jax.Array
    posterior_covariance: jax.Array
    revision: jax.Array
    revision_covariance: jax.Array
    transmission: jax.Array


class ReliableUpdate(NamedTuple):
    """Moments of an informative/null-source posterior mixture."""

    probability: jax.Array
    mean: jax.Array
    covariance: jax.Array
    informative: Update


def sym(array: jax.Array) -> jax.Array:
    """Remove antisymmetric floating-point roundoff only."""

    array = jnp.asarray(array)
    return (array + array.T) / 2


def logdet_spd(array: jax.Array) -> jax.Array:
    """Return the log determinant of a strictly positive-definite matrix."""

    chol = jnp.linalg.cholesky(jnp.asarray(array))
    return 2 * jnp.log(jnp.diag(chol)).sum()


def validate_channel(m, prior_covariance, measurement, signal, noise_covariance, tol=1e-10) -> None:
    """Validate a Gaussian channel without changing any supplied value.

    The prior covariance may be singular. The measurement-noise covariance
    must be strictly positive definite. Small negative prior eigenvalues are
    tolerated only at the scale specified by ``tol``; inputs are never
    repaired or regularized.
    """

    m, prior_covariance, measurement, signal, noise_covariance = map(
        np.asarray, (m, prior_covariance, measurement, signal, noise_covariance)
    )
    if m.ndim != 1 or signal.ndim != 1:
        raise ValueError("mean and observation must be vectors")
    n, r = len(m), len(signal)
    if r == 0:
        raise ValueError("the observation must contain at least one component")
    if prior_covariance.shape != (n, n):
        raise ValueError("prior covariance has an incompatible shape")
    if measurement.shape != (r, n):
        raise ValueError("measurement matrix has an incompatible shape")
    if noise_covariance.shape != (r, r):
        raise ValueError("noise covariance has an incompatible shape")
    for array in (m, prior_covariance, measurement, signal, noise_covariance):
        if not np.isfinite(array).all():
            raise ValueError("non-finite channel input")
    for covariance in (prior_covariance, noise_covariance):
        if not np.allclose(covariance, covariance.T, atol=tol, rtol=tol):
            raise ValueError("covariance must be symmetric")
    scale = max(1.0, float(np.linalg.norm(prior_covariance, 2)))
    if np.linalg.eigvalsh(prior_covariance).min() < -tol * scale:
        raise ValueError("prior covariance is not positive semidefinite")
    try:
        np.linalg.cholesky(noise_covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("noise covariance must be positive definite") from exc


def condition(m, prior_covariance, measurement, signal, noise_covariance) -> Update:
    """Condition ``z ~ N(m, P)`` on ``signal = measurement @ z + noise``.

    A Cholesky solve is used for the gain and evidence. The posterior uses the
    Joseph form. ``information_nats`` is ex-ante mutual information, whereas
    ``log_evidence`` is realization dependent.
    """

    m, prior_covariance, measurement, signal, noise_covariance = map(
        jnp.asarray, (m, prior_covariance, measurement, signal, noise_covariance)
    )
    innovation = signal - measurement @ m
    innovation_covariance = sym(measurement @ prior_covariance @ measurement.T + noise_covariance)
    chol = jnp.linalg.cholesky(innovation_covariance)
    gain = cho_solve((chol, True), measurement @ prior_covariance).T
    identity = jnp.eye(prior_covariance.shape[0], dtype=prior_covariance.dtype)
    residual_map = identity - gain @ measurement
    posterior_covariance = sym(
        residual_map @ prior_covariance @ residual_map.T + gain @ noise_covariance @ gain.T
    )
    whitened_loading = solve_triangular(chol, measurement @ prior_covariance, lower=True)
    reduction = sym(whitened_loading.T @ whitened_loading)
    logdet_innovation = 2 * jnp.log(jnp.diag(chol)).sum()
    log_evidence = -0.5 * (
        signal.size * jnp.log(2 * jnp.pi)
        + logdet_innovation
        + innovation @ cho_solve((chol, True), innovation)
    )
    information_nats = 0.5 * (logdet_innovation - logdet_spd(noise_covariance))
    return Update(
        m + gain @ innovation,
        posterior_covariance,
        gain,
        innovation,
        innovation_covariance,
        reduction,
        information_nats,
        log_evidence,
    )


def linear_payoff(m, prior_covariance, update: Update, loading, noise_covariance, offset) -> Payoff:
    """Transport a state update to ``Y = loading @ z + offset + noise``."""

    m, prior_covariance, loading, noise_covariance, offset = map(
        jnp.asarray, (m, prior_covariance, loading, noise_covariance, offset)
    )
    prior_mean = loading @ m + offset
    posterior_mean = loading @ update.mean + offset
    return Payoff(
        prior_mean,
        posterior_mean,
        sym(loading @ prior_covariance @ loading.T + noise_covariance),
        sym(loading @ update.covariance @ loading.T + noise_covariance),
        posterior_mean - prior_mean,
        sym(loading @ update.reduction @ loading.T),
        loading @ update.gain,
    )


def weighted_value(reduction, weight) -> jax.Array:
    """Reduction of optimal quadratic prediction loss: ``trace(weight @ reduction)``."""

    return jnp.trace(jnp.asarray(weight) @ jnp.asarray(reduction))


def payoff_information(prior_covariance, posterior_covariance) -> jax.Array:
    """Return ``I(Y; S)`` for a jointly Gaussian pair with SPD payoff covariances."""

    return 0.5 * (logdet_spd(prior_covariance) - logdet_spd(posterior_covariance))


def cara_information_fee(prior_covariance, posterior_covariance, risk_aversion) -> jax.Array:
    """Ex-ante CARA-Gaussian cash indifference fee for a private signal.

    This formula assumes unrestricted positions, fixed executable prices, no
    market impact, no random endowment, and strictly positive risk aversion.
    It is not the equilibrium price impact of public news.
    """

    return payoff_information(prior_covariance, posterior_covariance) / jnp.asarray(risk_aversion)


def quadratic_decision_value(payoff_reduction, cost_matrix) -> jax.Array:
    """Unconstrained decision value under a fixed SPD quadratic cost matrix."""

    return 0.5 * jnp.trace(jnp.linalg.solve(jnp.asarray(cost_matrix), jnp.asarray(payoff_reduction)))


def normal_interval(mean, variance, coverage=0.95) -> jax.Array:
    """Central scalar Gaussian predictive interval, including zero variance."""

    if not 0 < coverage < 1:
        raise ValueError("coverage must lie in (0, 1)")
    radius = ndtri((1 + coverage) / 2) * jnp.sqrt(jnp.asarray(variance))
    return jnp.stack((jnp.asarray(mean) - radius, jnp.asarray(mean) + radius))


def normal_es(mean, variance, alpha=0.95) -> jax.Array:
    """Upper-tail expected shortfall of a scalar Gaussian loss."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    z = ndtri(alpha)
    return jnp.asarray(mean) + jnp.sqrt(jnp.asarray(variance)) * jnp.exp(-z * z / 2) / (
        jnp.sqrt(2 * jnp.pi) * (1 - alpha)
    )


def propagate(m, prior_covariance, transitions, process_covariances, biases):
    """Propagate Gaussian moments through chronological affine transitions.

    Inputs have shapes ``(h,n,n)``, ``(h,n,n)``, and ``(h,n)``. The returned
    tuple contains future mean, future covariance, ordered propagator, and the
    covariance contributed by future process noise. Empty leading arrays
    implement the horizon-zero identity.
    """

    m, prior_covariance, transitions, process_covariances, biases = map(
        jnp.asarray, (m, prior_covariance, transitions, process_covariances, biases)
    )

    def step(carry, inputs):
        mean, covariance, propagator, future_noise = carry
        transition, process_covariance, bias = inputs
        next_carry = (
            transition @ mean + bias,
            sym(transition @ covariance @ transition.T + process_covariance),
            transition @ propagator,
            sym(transition @ future_noise @ transition.T + process_covariance),
        )
        return next_carry, None

    n = prior_covariance.shape[0]
    initial = (
        m,
        prior_covariance,
        jnp.eye(n, dtype=prior_covariance.dtype),
        jnp.zeros_like(prior_covariance),
    )
    return lax.scan(step, initial, (transitions, process_covariances, biases))[0]


def residualize_second(measurement1, measurement2, signal1, signal2, noise11, noise21, noise22):
    """Residualize a second packet against correlated first-packet noise."""

    measurement1, measurement2, signal1, signal2, noise11, noise21, noise22 = map(
        jnp.asarray, (measurement1, measurement2, signal1, signal2, noise11, noise21, noise22)
    )
    coefficient = jnp.linalg.solve(noise11, noise21.T).T
    return (
        measurement2 - coefficient @ measurement1,
        signal2 - coefficient @ signal1,
        sym(noise22 - coefficient @ noise21.T),
    )


def cashflow_projection(transitions, process_covariances, biases, cashflow_weights, discounts):
    """Project a discounted multi-horizon cash-flow sum onto the current state.

    Returns the time-zero loading, deterministic offset, and independent future
    process-noise variance. The backward recursion preserves cross-horizon
    dependence that would be lost by summing marginal variances.
    """

    transitions, process_covariances, biases, cashflow_weights, discounts = map(
        jnp.asarray, (transitions, process_covariances, biases, cashflow_weights, discounts)
    )
    discounted_weights = cashflow_weights * discounts[:, None]
    loading = discounted_weights[-1]
    offset = jnp.asarray(0.0, dtype=discounted_weights.dtype)
    variance = jnp.asarray(0.0, dtype=discounted_weights.dtype)
    for index in range(transitions.shape[0] - 1, -1, -1):
        variance = variance + loading @ process_covariances[index] @ loading
        offset = offset + loading @ biases[index]
        loading = discounted_weights[index] + transitions[index].T @ loading
    return loading, offset, variance


def quadratic_moments(m, prior_covariance, linear, quadratic, offset=0.0, independent_variance=0.0):
    """Exact moments of ``offset + linear.T z + z.T quadratic z / 2 + noise``.

    ``quadratic`` must be symmetric. These moments do not imply that the payoff
    distribution is Gaussian.
    """

    m, prior_covariance, linear, quadratic = map(jnp.asarray, (m, prior_covariance, linear, quadratic))
    mean = offset + linear @ m + 0.5 * (m @ quadratic @ m + jnp.trace(quadratic @ prior_covariance))
    gradient_at_mean = linear + quadratic @ m
    variance = (
        gradient_at_mean @ prior_covariance @ gradient_at_mean
        + 0.5 * jnp.trace(quadratic @ prior_covariance @ quadratic @ prior_covariance)
        + independent_variance
    )
    return mean, variance


def reliability_update(
    m,
    prior_covariance,
    measurement,
    signal,
    noise_covariance,
    prior_probability,
    null_mean,
    null_covariance,
) -> ReliableUpdate:
    """Exact informative/null-source posterior-mixture moments.

    With probability ``prior_probability`` the signal follows the informative
    Gaussian channel; otherwise it follows a state-independent Gaussian null
    model. The returned covariance is the exact mixture covariance, not a
    Gaussian approximation to the posterior distribution.
    """

    update = condition(m, prior_covariance, measurement, signal, noise_covariance)
    signal, null_mean, null_covariance = map(jnp.asarray, (signal, null_mean, null_covariance))
    residual = signal - null_mean
    chol = jnp.linalg.cholesky(null_covariance)
    null_log_evidence = -0.5 * (
        signal.size * jnp.log(2 * jnp.pi)
        + logdet_spd(null_covariance)
        + residual @ cho_solve((chol, True), residual)
    )
    probability = jnp.asarray(prior_probability)
    log_weights = jnp.stack(
        (
            jnp.log(probability) + update.log_evidence,
            jnp.log1p(-probability) + null_log_evidence,
        )
    )
    posterior_probability = jnp.exp(log_weights[0] - logsumexp(log_weights))
    m = jnp.asarray(m)
    prior_covariance = jnp.asarray(prior_covariance)
    delta = update.mean - m
    covariance = sym(
        posterior_probability * update.covariance
        + (1 - posterior_probability) * prior_covariance
        + posterior_probability * (1 - posterior_probability) * jnp.outer(delta, delta)
    )
    return ReliableUpdate(posterior_probability, m + posterior_probability * delta, covariance, update)


def ensemble_weights(prior_weights, log_likelihoods) -> jax.Array:
    """Bayes-reweight fixed parameter or regime particles after a packet."""

    log_weights = jnp.log(jnp.asarray(prior_weights)) + jnp.asarray(log_likelihoods)
    return jnp.exp(log_weights - logsumexp(log_weights))


def edge_attribution(transition, baseline, revision, exposure, horizon: int) -> jax.Array:
    """Integrated gradients of ``exposure.T @ transition**horizon @ revision``.

    The attribution holds the revision and exposure fixed. It depends on the
    chosen state basis and baseline. Gauss-Legendre order ``ceil(horizon/2)``
    integrates the polynomial exactly in exact arithmetic.
    """

    if not isinstance(horizon, int) or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")
    transition, baseline, revision, exposure = map(jnp.asarray, (transition, baseline, revision, exposure))
    if horizon == 0:
        return jnp.zeros_like(transition)
    nodes, weights = np.polynomial.legendre.leggauss(max(1, (horizon + 1) // 2))
    nodes = jnp.asarray((nodes + 1) / 2)
    weights = jnp.asarray(weights / 2)

    def response(matrix):
        return exposure @ jnp.linalg.matrix_power(matrix, horizon) @ revision

    gradient = jax.grad(response)
    path = baseline + nodes[:, None, None] * (transition - baseline)
    gradients = jax.vmap(gradient)(path)
    return (transition - baseline) * jnp.einsum("k,kij->ij", weights, gradients)


class Relevance(NamedTuple):
    """Canonical-correlation diagnostics of a payoff revision (Theorem 3.3)."""

    squared_correlations: jax.Array
    information_nats: jax.Array
    trace: jax.Array
    lower_bound: jax.Array
    upper_bound: jax.Array


def canonical_relevance(prior_covariance, revision_covariance) -> Relevance:
    """Eigenvalues of ``J = Sigma^{-1/2} Delta Sigma^{-1/2}`` and their bounds.

    ``prior_covariance`` is the SPD payoff covariance and ``revision_covariance``
    the PSD covariance of the forecast revision, so the posterior payoff
    covariance ``prior - revision`` must also be SPD. The returned bounds are
    ``1 - exp(-2 I) <= trace(J) <= p (1 - exp(-2 I / p))``.
    """

    prior_covariance, revision_covariance = map(jnp.asarray, (prior_covariance, revision_covariance))
    chol = jnp.linalg.cholesky(prior_covariance)
    left = solve_triangular(chol, revision_covariance, lower=True)
    whitened = sym(solve_triangular(chol, left.T, lower=True))
    squared_correlations = jnp.linalg.eigvalsh(whitened)
    information = -0.5 * jnp.log1p(-squared_correlations).sum()
    dimension = squared_correlations.shape[0]
    return Relevance(
        squared_correlations,
        information,
        squared_correlations.sum(),
        -jnp.expm1(-2 * information),
        -dimension * jnp.expm1(-2 * information / dimension),
    )


def whitened_transmission(update: Update, loading) -> jax.Array:
    """Return ``L_h = B K L`` for the Cholesky factor ``L`` of the innovation covariance.

    The whitened innovation ``L^{-1}(s - M m)`` is standard normal and the
    payoff revision equals ``L_h`` applied to it. Singular values of ``L_h``
    do not depend on the chosen square root (Corollary 3.2); individual
    columns do.
    """

    chol = jnp.linalg.cholesky(update.innovation_covariance)
    return jnp.asarray(loading) @ update.gain @ chol


def noise_sensitivity(prior_covariance, measurement, noise_covariance, weight, perturbation) -> jax.Array:
    """Directional derivative of ``trace(W D(R))`` along ``R + eps E`` (Proposition 3.4).

    Equals ``-trace(Sigma^{-1} M P W P M.T Sigma^{-1} E)``, which is
    nonpositive when ``W`` and ``E`` are positive semidefinite.
    """

    prior_covariance, measurement, noise_covariance, weight, perturbation = map(
        jnp.asarray, (prior_covariance, measurement, noise_covariance, weight, perturbation)
    )
    innovation_covariance = sym(measurement @ prior_covariance @ measurement.T + noise_covariance)
    chol = jnp.linalg.cholesky(innovation_covariance)
    loading = cho_solve((chol, True), measurement @ prior_covariance)
    return -jnp.trace(loading @ weight @ loading.T @ perturbation)


def cara_positions(mean, covariance, risk_aversion) -> jax.Array:
    """Unrestricted CARA-Gaussian optimal positions ``Sigma^{-1} mu / gamma`` (Theorem 4.2).

    Pass the prior moments for the no-signal position and the conditional
    moments ``(mu_S, Sigma_+)`` for the position after observing the signal.
    """

    chol = jnp.linalg.cholesky(jnp.asarray(covariance))
    return cho_solve((chol, True), jnp.asarray(mean)) / jnp.asarray(risk_aversion)


def mean_error_regret_bound(cost_matrix, true_mean, used_mean) -> jax.Array:
    """Regret bound ``(mu_hat - mu).T H^{-1} (mu_hat - mu) / 2`` of Theorem 4.1.

    It bounds the loss from optimizing with a mistaken mean over any closed
    convex action set and is attained for unrestricted positions.
    """

    error = jnp.asarray(used_mean) - jnp.asarray(true_mean)
    return 0.5 * error @ jnp.linalg.solve(jnp.asarray(cost_matrix), error)


def structural_response(transition, control, control_input, horizon: int) -> jax.Array:
    """State response ``F**(h-1) G u`` to an identified structural input (Section 2.2).

    This is an intervention effect in ``z_{t+1} = F z_t + G u_t + w_{t+1}``.
    It is a different object from a measurement revision ``Phi_h K (s - M m)``;
    an observational state-space model does not identify it.
    """

    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    transition, control, control_input = map(jnp.asarray, (transition, control, control_input))
    return jnp.linalg.matrix_power(transition, horizon - 1) @ control @ control_input


def from_filtered(filtered, time_index: int, measurement, signal, noise_covariance) -> Update:
    """Assimilate a genuinely new packet into a Dynamax filtered marginal.

    The selected filtered marginal must use only observations available before
    the packet. Do not assimilate the same observation twice and do not pass a
    smoothed marginal containing future information.
    """

    return condition(
        filtered.filtered_means[time_index],
        filtered.filtered_covariances[time_index],
        measurement,
        signal,
        noise_covariance,
    )
