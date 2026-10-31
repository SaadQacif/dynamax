# fininfo: decision-aware financial information transport

A research prototype built on Dynamax. It maps an information packet, modelled
as a statistical experiment with a likelihood rather than as a similarity score,
to:

- a revised distribution of financial outcomes,
- a decision value with explicit units, and
- an attribution that respects dependence and identifiability.

It implements the draft *Decision-Aware Financial Information Transport in
State-Space Models* (v2).

All measured results in this package are **synthetic**. Several things remain
open empirical work: semantic likelihood learning, real-market calibration and
causal identification. Classical results are credited, not presented as new
theorems.

## Layout

| File | Purpose |
| --- | --- |
| `core.py` | Pure JAX kernels: Gaussian update, payoff transport, information and decision values, propagation, residualization, cash flows, reliability mixtures, edge attribution, Dynamax adapter |
| `algorithms.py` | Eager NumPy/SciPy reference code: coalition values, Shapley, discounted Gramian, mixture VaR/ES, entropic projection, conformal radius |
| `experiments.py` | Regenerates the Section 6 coverage studies, `results/results.json` and `results/coverage.png` |
| `tests/` | `test_core.py`, `test_theory.py`, `test_extensions.py` and `test_dynamax.py` (runs against real `LinearGaussianSSM.filter` output) |

```bash
pip install -e . scipy pytest matplotlib
python -m pytest fininfo            # 64-bit arithmetic is enabled in tests/conftest.py
python -m fininfo.experiments
```

Applications must enable `jax_enable_x64` themselves. `validate_channel` is an
eager check. Compiled kernels perform no data-dependent checks, jitter or
clipping.

## Probability model

Everything conditions on the information set $\mathcal F_{t-}$ just before the
packet arrives:

$$Z \sim \mathcal N(m,P),\qquad S = MZ+\varepsilon,\qquad \varepsilon\sim\mathcal N(0,R),\ R\succ0,\ \varepsilon\perp Z .$$

$R$ may have cross-packet blocks. $M$ and $R$ are fixed before $S$ is seen.
Future dynamics are

$$z_{t+j}=F_j z_{t+j-1}+a_j+w_j .$$

The payoff is

$$Y_h = B_h Z + d_h + \eta_h,\qquad B_h = C_h\Phi_h,\qquad \Phi_h=F_h\cdots F_1 .$$

Here $\eta_h \sim \mathcal N(0,N_h)$ is independent of $(Z,\varepsilon)$ and
includes future process noise (`propagate` returns
$\Phi_h$ and $\sum_j (F_h\cdots F_{j+1})Q_j(\cdot)^\top$).

| Quantity | Meaning | Units | Function |
| --- | --- | --- | --- |
| $I(Z;S)$ | ex-ante state information | nats | `condition(...).information_nats` |
| $\delta_h(s)$ | realized revision of the payoff mean | payoff | `linear_payoff(...).revision` |
| $V_W=\operatorname{tr}(W\Delta_h)$ | expected reduction of squared forecast loss | payoff² | `weighted_value` |
| $c_{\text{info}}$ | cash fee in an explicit CARA problem | currency | `cara_information_fee` |
| $\mathrm{ES}_\alpha(L\mid s)$ | physical-model tail loss | loss | `normal_es`, `mixture_es` |

**Belief revision is not intervention (Section 2.2).** A structural input moves
the state by $F^{h-1}Gu$ (`structural_response`). A measurement moves the
forecast by $\Phi_hK(s-Mm)$. An observational state-space model does not
identify the former.

## Exact Gaussian transport (Section 3)

Define

$$\Sigma = MPM^\top+R,\qquad K = PM^\top\Sigma^{-1},\qquad D = PM^\top\Sigma^{-1}MP,\qquad P_+=P-D .$$

**Theorem 3.1.** The posterior is $Z\mid s\sim\mathcal N\big(m+K(s-Mm),\,P_+\big)$.
This holds even when $P$ is singular.

With $T_h = B_hK$, the revision is

$$\delta_h(S)=T_h(S-Mm)\sim\mathcal N(0,\Delta_h),\qquad \Delta_h=B_hDB_h^\top .$$

For a PSD weight $A$, the ex-ante reduction of optimal quadratic loss is
$\operatorname{tr}(A\Delta_h)$. `condition` uses a Cholesky gain and the Joseph
covariance form.

**Corollary 3.2 (whitening and invariance).** Let $\Sigma=LL^\top$. Then
$\delta_h=L_hu$ with $u\sim\mathcal N(0,I)$ and $L_h=B_hKL$
(`whitened_transmission`). The singular values of $L_h$ do not depend on the
choice of square root. The revision, $\Delta_h$ and $I(Y_h;S)$ are invariant
under invertible changes of state or signal coordinates.

**Theorem 3.3 (information and canonical modes).** State information is

$$I(Z;S)=\tfrac12\log\det\Sigma/\det R .$$

Let $\rho_i^2$ be the eigenvalues of $J=\Sigma_h^{-1/2}\Delta_h\Sigma_h^{-1/2}$.
Then

$$I(Y_h;S) = -\tfrac12\sum_i\log(1-\rho_i^2)\ \le\ I(Z;S),$$

and

$$1-e^{-2I(Y_h;S)}\le\operatorname{tr}J\le p\big(1-e^{-2I(Y_h;S)/p}\big)$$

(`canonical_relevance`). State information can be large while every
financially relevant mode is zero.

**Proposition 3.4 (noise sensitivity).** For $E\succeq0$,

$$\tfrac{d}{d\epsilon}\operatorname{tr}\big(WD(R+\epsilon E)\big)\big|_0=-\operatorname{tr}\big(\Sigma^{-1}MPWPM^\top\Sigma^{-1}E\big)\le0$$

(`noise_sensitivity`).

## Decisions and monetary value (Section 4)

**Theorem 4.1.** Take positions $a\in\mathcal A$, a closed convex set, with
objective $a^\top\mu-\tfrac12a^\top Ha$ and fixed $H\succ0$ in currency. Then

$$0\le\mathbb E f(\mu_S)-f(\mu)\le\tfrac12\operatorname{tr}(H^{-1}\Delta),$$

with equality for unrestricted positions (`quadratic_decision_value`). Deciding
with a mistaken mean $\hat\mu$ costs at most
$\tfrac12(\hat\mu-\mu)^\top H^{-1}(\hat\mu-\mu)$ (`mean_error_regret_bound`).

**Theorem 4.2 (CARA-Gaussian fee).** Assume:

- utility $-e^{-\gamma w}$;
- unrestricted positions;
- fixed executable prices;
- a private signal.

Then the unique ex-ante indifference fee is

$$c_{\text{info}}=\frac{1}{2\gamma}\log\frac{\det\Sigma}{\det\Sigma_+}=\frac{I(Y;S)}{\gamma}.$$

The optimal positions are $\Sigma^{-1}\mu/\gamma$ and
$\Sigma_+^{-1}\mu_S/\gamma$ (`cara_positions`). This prices access to the
signal for this investor. It is not a risk-neutral price and not the
equilibrium response to public news.

## Dependence, attribution and uncertain evidence (Section 5)

- **Correlated packets.** Residualizing with $A=R_{21}R_{11}^{-1}$ gives
  $\tilde S_2=S_2-AS_1$ with noise $R_{22}-R_{21}R_{11}^{-1}R_{12}$. Sequential
  updating is then exactly equal to the batch update (`residualize_second`).
  Ignoring $R_{21}$ is not equivalent.
- **No diminishing returns.** Let $v(A)=\operatorname{tr}[W(P-P_A)]$. This is
  monotone, and its Shapley allocation is nonnegative and efficient
  (`coalition_values`, `shapley`). However, with $Y=Z_1$,
  $S_1=Z_1+Z_2+\varepsilon_1$ and $S_2=Z_2+\varepsilon_2$, we get
  $v(\{2\})=0<v(\{1,2\})-v(\{1\})$. The objective is not submodular.
- **Edge attribution.** Integrated gradients of $c^\top F^h d$ with respect to
  the entries of $F$ sum exactly to $r(F)-r(F_0)$, using Gauss–Legendre order
  $\lceil h/2\rceil$ (`edge_attribution`). The result depends on the basis and
  the baseline. Latent wires have no intrinsic economic identification.
- **Discounted cash flows.** The backward recursion
  $b_{j-1}=d_{j-1}c_{j-1}+F_j^\top b_j$ gives the loading $b_0$, the offset,
  and the future-noise variance $\sum_jb_j^\top Q_jb_j$
  (`cashflow_projection`). This keeps cross-horizon dependence that summing
  marginal variances would lose.
- **Reliability.** Suppose the source is informative with probability $\pi$
  and otherwise produces null evidence. The posterior is a mixture with mean
  $m+wd$ and covariance $wP_++(1-w)P+w(1-w)dd^\top$
  (`reliability_update`). It can exceed the prior variance.
- **Quadratic payoffs.** `quadratic_moments` gives exact moments. The law is
  not Gaussian.

## Synthetic evidence (Section 6)

These are nominal 95% predictive-interval coverages from
`python -m fininfo.experiments`, with seeds recorded in
`results/results.json`:

| Study | Correct model | Misspecified model |
| --- | --- | --- |
| 100,000 draws, packet errors correlated 0.8 | full $R$: 94.97% | diagonal $R$: 91.15% |
| 80,000 draws, informative probability 0.6, null $\mathcal N(0,4)$ | reliability mixture: 95.13% | always trusted: 72.54% |

The draft does not specify its exact generating parameters, so these
percentages differ from the draft's figures. The qualitative conclusion
reproduces: ignoring dependence or unreliability breaks calibration.

## What remains open

The semantic-to-likelihood map is untrained. A real-data study must:

- freeze the extractor before evaluation;
- keep publication and availability timestamps;
- model shared sources;
- separate training periods from future test periods;
- report coverage and decision regret against no-news and simpler baselines.

Only filtered (never smoothed) marginals may feed `from_filtered` for live
forecasting. Entropic projection (`entropy_project`) is a scenario projection,
not a dynamic no-arbitrage pricing model. Physical and pricing probabilities
must stay distinct.

## References

Särkkä & Svensson, *Bayesian Filtering and Smoothing* (2023) · Cover & Thomas,
*Elements of Information Theory* (2006) · Shapley (1952) · Krause, Singh &
Guestrin, JMLR 9 (2008) · Sundararajan, Taly & Yan, ICML (2017) · Meucci, *Fully
flexible views* (2008) · Brody & Law, *Theory of information pricing* (2011) ·
Grossman & Stiglitz, AER 70(3) (1980) · Angelopoulos & Bates (2021).
