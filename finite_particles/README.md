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
(matching `entropic_potential.idp`'s defaults). Both files hard-code a
single effective dimension $d$ rather than taking it as a sweep parameter:
`gif.jl` uses $d=5$, `l1_decay.jl` uses $d=2$. `dUdx` is written with an
explicit `d==2` special case (dropping the $(1+\kappa_0 g(x))^{d-2}$ factor,
which is 1 at $d=2$ anyway) alongside the general-$d$ formula used for
`gif.jl`'s $d=5$ — both branches agree with `dUdx` in
`entropic_potential.idp` by direct substitution.

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

Short, deliberately under-converged runs (few steps) of the same four
regimes (Unbiased, ABF $\gamma=0.5$, ABP $\alpha=1$, ABP $\alpha=10$),
rendered as a $2\times2$ animated GIF of particles moving over a filled
contour of $U$, so the initial pile-up at $x_0=-1$ and the early transient
are visible rather than the converged steady state. Uses a wider $y$-box
($[-6,6]$ in `main()`, vs. $\pm2.5$ in `../edp/`) than the PDE domain —
a visualization-only choice (no data is compared against FreeFEM output
here), presumably to avoid visible boundary reflection artifacts while
$d=5$ leaves $y$ weakly confined near $x=-1$.

Output: `particles_2d.gif`.

**Note (not fixed, flagging for you):** `make_gif` always writes to the
literal filename `"particles_2d.gif"` regardless of the `const d` used in
the run — it doesn't encode `d` in the output name. `particles_10d.gif`
in this directory must have been produced by an earlier run (`d=10`?) and
saved/renamed by hand; with the current `const d = 5`, re-running will
overwrite `particles_2d.gif` with $d=5$ data under a filename that says
"2d". Say the word if you'd like the output name parameterized by `d`
(e.g. `particles_$(d)d.gif`) — I left this alone since it's a behavior
change, not a doc fix.

## 5. Files

| file | role |
|---|---|
| `gif.jl` | 4-panel animated GIF, $d=5$, GR backend. |
| `l1_decay.jl` | $\alpha$-sweep, $L^1$-to-stationary-marginal metric, $d=2$, PlotlyJS backend, threaded over $(\alpha,\text{rep})$. |

```
julia gif.jl
julia --threads=32 l1_decay.jl
```
