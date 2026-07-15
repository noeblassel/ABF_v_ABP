# Interacting particle system (finite $N$)

The actual enhanced-sampling algorithms, run as a genuine finite-population
interacting particle system: unlike `../mean_field/`, the bias at each step
is estimated **on-line from the ensemble itself** (a running mean-force
average for ABF, an instantaneous kernel density estimate for ABP) — no
precomputed FreeFEM data is read anywhere in this directory. This is the
`IPS/` tier described in the top-level `README.md` (the `../ips/` folder is
currently an empty placeholder for it; these two scripts are that
implementation). Implementation in Julia, figures via Plots.jl.

## 1. Model

Same reduced $(x,y)$ potential as `../edp/` and `../mean_field/`:

```math
U(x,y) = (x^2-1)^2 + k(x)^{d-1}\,y^2, \qquad k(x) = 1+\kappa_0\, e^{-x^2/2\sigma^2},
```

with $\kappa_0=1$, $\sigma=0.25$, $\beta=1$, domain $[-2.5,2.5]$ in $x$
(matching `entropic_potential.idp`'s defaults). `gif.jl` takes $d$ as a
runtime `Params` field (like `../mean_field/gif.jl`) and sweeps
$d\in\{2,10\}$; `l1_decay.jl` still hard-codes $d=2$ via `const d`, with
`dUdx` written as an explicit `d==2` special case (dropping the
$(1+\kappa_0 g(x))^{d-2}$ factor, which is 1 at $d=2$ anyway) — this
branch agrees with `dUdx` in `entropic_potential.idp` by direct
substitution, as does `gif.jl`'s general-$d$ formula.

## 2. Algorithms

Both files add the bias only along $x$; $y$ always evolves under its own
unbiased force. Using the same sign convention verified in
`../mean_field/README.md` (drift $=-(\partial_xU+g)$, i.e. $g$ is
subtracted):

- **ABF** — a running (all-time, since $t=0$) bin-average of the sampled
  mean force $\partial_xU(X_t,Y_t)$ gives $\widehat{A'}(x)$; the bias
  applied is $\gamma\widehat{A'}(x)$ with $\gamma=\alpha/(1+\alpha)$,
  i.e. drift $=-\partial_xU+\gamma\widehat{A'}(x)$ — matches
  $g_{\rm ABF}=-\gamma A'(x)$ from `../edp/README.md` exactly, with
  $A'(x)=\mathbb E[\partial_xU\mid X=x]$ estimated empirically instead of
  from the PDE marginal.
- **ABP** — a Gaussian-kernel density estimate $\hat\rho^1(x)$ of the
  ensemble's instantaneous $x$-positions (bandwidth `sigma_kernel`,
  rebuilt from scratch every step, unlike ABF's cumulative average — this
  matches the algorithms' intended distinction: ABF averages the force
  over history, ABP reacts to the *instantaneous* population density);
  the bias applied is $\alpha\,\partial_x\bigl(-\beta^{-1}\ln\hat\rho^1(x)\bigr)
  = -(\alpha/\beta)\,\partial_x\ln\hat\rho^1(x)$ — matches
  $g_{\rm ABP}=(\alpha/\beta)\partial_x\ln\rho^1$ exactly, with $\rho^1$
  replaced by the finite-$N$ KDE.

## 3. `l1_decay.jl` — L1 convergence sweep

Computes, for Unbiased / ABP($\alpha$) / ABF($\gamma=\alpha/(1+\alpha)$),
the $L^1$ distance between a KDE of the ensemble's $x$-marginal and the
theoretical stationary marginal

```math
\rho^1_\infty(x)\ \propto\ e^{\beta\gamma F(x)}\int_{y_{\min}}^{y_{\max}} e^{-\beta U(x,y)}\,{\rm d}y,
```

evaluated by numerical quadrature over the **same finite, reflecting
$y$-domain the dynamics uses** (rather than assuming an infinite
transverse extent), so the reference is self-consistent with the
truncated-box simulation. The free-energy term used,
`a(x) = W(x) - log(4π/(β(1+κg(x))))/(2β)`, is algebraically
$W(x)+\tfrac{1}{2\beta}\ln k(x)$ — exactly `entropic_potential.idp`'s `Fx`
at $d=2$, the dimension this file fixes.

Sweeps $\alpha\in\{0.1,0.2,0.5,1,2,5,10,20,50\}$, each curve averaged over
`N_REPS` replicates, plus fixed references (unbiased, $\gamma=1$).
This $\alpha$ grid is independent of `../edp/`'s/`../mean_field/`'s
$\{0.125,\dots,16\}$ grid — no data is shared between them, so there is no
compatibility requirement on the grid values themselves, only on the model
and bias formulas above.

Output: `alpha_sweep_l1_vs_time_2d.html` (2 panels: ABP($\alpha$),
ABF($\gamma$), escape fraction past $x=0$ in each legend entry).

## 4. `gif.jl` — animated visualization

Finite-particle counterpart of `../mean_field/gif.jl`: same potential,
same $\pm2.5$ domain in both $x$ and $y$, same Gaussian initial condition
($x_0\sim\mathcal N(-1,0.05^2)$, $y_0\sim\mathcal N(0,0.05^2)$), same
early-transient window (`SIM_TMAX`-equivalent $=0.5$, `dt=5e-4`, so
1000 steps), same dense frame schedule (`early_phase_steps=400,
save_every_early=4, save_every=32`, `fps=100`) — but here ABF and ABP
build their bias **on-line** from the $N$-particle ensemble (the
estimators from §2) instead of reading a precomputed FreeFEM bias field.
Any visible difference from `../mean_field/gif.jl`'s animation at matched
$N$, $d$, and $\alpha$ is finite-population estimation noise, not a
different model.

For each $d\in\{2,10\}$, renders a $2\times4$ grid matching
`../mean_field/gif.jl`'s layout exactly:

```
row 1:  ABP(α=0.125) | ABP(α=1) | ABP(α=4) | ABP(α=16)
row 2:  Unbiased     | ABF(α=1) | ABF(α=4) | ABF(α=16)
```

ABF's bias uses $\gamma=\alpha/(1+\alpha)$ at $\alpha=1,4,16$ — the same
$\alpha$ values ABP uses — so panels are a matched-$\alpha$ comparison of
the two algorithms.

Output: `particles_d2.gif`, `particles_d10.gif` (the output name is now
parameterized by `d`, fixing the earlier hardcoded-filename issue where
`make_gif` always wrote to the literal `"particles_2d.gif"` regardless of
which `d` was actually run).

## 5. Files

| file | role |
|---|---|
| `gif.jl` | $2\times4$ animated GIF, $d\in\{2,10\}$, GR backend. |
| `l1_decay.jl` | $\alpha$-sweep, $L^1$-to-stationary-marginal metric, $d=2$, PlotlyJS backend, threaded over $(\alpha,\text{rep})$. |

```
julia gif.jl
julia --threads=32 l1_decay.jl
```
