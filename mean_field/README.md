# Mean-field (McKean–Vlasov) particle simulation

Monte Carlo integration of the mean-field limit of the ABF / ABP algorithms: a
single representative replica driven by a bias that is *not* estimated
on-line from the replica population, but read from precomputed bias fields
produced by the FreeFEM density solver in `../edp/` (see the top-level
`README.md` for the physical model and `../edp/README.md` for the
discretization that generated these bias fields). Implementation in Julia
(`main.jl`), figures via Plots.jl/PlotlyJS.

## 1. Model

Same reduced $(x,y)$ potential as `../edp/`, with $x$ the reaction coordinate
and $y$ the transverse "gate" coordinate:

```math
U(x,y) = (x^2-1)^2 + k(x)^{d-1}\,y^2, \qquad k(x) = 1+\kappa_0\, e^{-x^2/2\sigma^2}.
```

Defaults $\kappa_0=1$, $\sigma=0.25$, $\beta=1$, domain $[-x_{\max},x_{\max}]^2$
with $x_{\max}=2.5$ — identical to the defaults in `entropic_potential.idp`.
$k(x)$ has a bump at $x=0$, tightly confining $y$ near the barrier and
letting it roam freely near the wells $x=\pm1$ (where $\sigma=0.25$ makes
$e^{-x^2/2\sigma^2}$ negligible).

## 2. Dynamics integrated

Each particle is a reflected overdamped Langevin path in the box
$[-x_{\max},x_{\max}]^2$:

```math
{\rm d}X_t = -\bigl(\partial_x U + g(X_t,t)\bigr)\,{\rm d}t + \sqrt{2\beta^{-1}}\,{\rm d}W_t,
\qquad
{\rm d}Y_t = -\partial_y U\,{\rm d}t + \sqrt{2\beta^{-1}}\,{\rm d}W_t,
```

reflected at $\pm x_{\max}$. $g(x,t)$ is the bias field: zero for the
**unbiased** run, and for **ABP**/**ABF** it is read from disk rather than
recomputed from an empirical marginal — this is the mean-field
approximation: the bias a single replica feels is the bias that the exact
(infinite-population) marginal law would produce, which is exactly what the
FreeFEM PDE solve represents. Concretely, $g$ is bilinearly interpolated in
$(x,t)$ from a precomputed grid:

```
biases/abp_alpha<alpha>_d<d>_bias.txt
biases/abf_alpha<alpha>_d<d>_bias.txt
```

Each file is $1001 \times 151$: rows $t=0,0.005,\dots,5$, columns $x$ on a
uniform grid over $[-2.5,2.5]$. Every entry is already the fully-formed bias
force (i.e. $\alpha$/$\gamma$ folded in) — no rescaling happens in
`main.jl`, the file value is used as-is (verified: the ABP row at $t=0$ is
antisymmetric about the initial mean $x=-1$ and scales exactly linearly with
$\alpha$ across files, matching $b_{\mathrm{ABP}}=(\alpha/\beta)\,\partial_x\ln\rho^1$
applied to a Gaussian centered at $-1$).

## 3. Initial condition

$x_0 \sim \mathcal N(-1, 0.05^2)$, $y_0 \sim \mathcal N(0, 0.05^2)$, i.i.d.
across particles — matching the Gaussian $\rho_0\propto
e^{-((x+1)^2+y^2)/2s_0^2}$, $s_0=0.05$, used to initialize the PDE solve
(`solver.idp`), so the bias fields are consistent with the state each
particle starts in.

## 4. Metric

For every particle we record the first-passage time to $x=0$ ($\tau_0$) and
to $x=1$ ($\tau_1$); since the path is continuous and $x_0=-1<0<1$,
$\tau_1\ge\tau_0$ always. We report, versus time, the cumulative fraction of
particles with $\tau_0\le t$ ("passed 0") and with $\tau_1\le t$ ("passed 0
then 1"), averaged over `N_REPS` independent replicates. This replaces an
earlier $L^1$-vs-target-density metric with something meaningful for a
finite population of independent, non-interacting mean-field replicas.

## 5. Consistency with the FreeFEM source of the bias fields

The bias files are the `*_bias.txt` output of `../edp/write_frames.idp`
(one row per stored frame, one column per node of the 1D marginal space
`Vh1`), run through `../edp/run_sweep.sh`. Everything `main.jl` assumes
about that data matches the FreeFEM defaults it was produced with:

| quantity | `main.jl` | `edp/` default | match |
|---|---|---|---|
| $\kappa_0$, $\sigma$, $\beta$ | `1.0, 0.25, 1.0` | `-kappa0 1 -sigma 0.25 -beta 1` | yes |
| potential $U$, $\partial_xU$, $\partial_yU$ | `U`, `dUdx`, `dUdy` | `entropic_potential.idp` (`U`, `dxU`, `dyU`) | yes, checked by hand |
| domain | $x,y\in[-2.5,2.5]$ | `-nx`: `square(nx,nx)` on $[-x_{\max},x_{\max}]^2$, $x_{\max}=2.5$ | yes |
| $x$-grid size | 151 columns | `Vh1.ndof = nx+1 = 151` for default `-nx 150` | yes |
| time grid | 1001 rows, $\Delta t_{\rm file}=0.005$, $T=5$ | default `-dt 0.005 -T 5 -stride 1` $\Rightarrow$ 1000 steps + initial frame = 1001 | yes |
| $\alpha$ grid | `0.125,0.25,0.5,1,2,4,8,16` | `run_sweep.sh` `SWEEP_ALPHA` default | yes |
| $d$ grid | files exist for $d=2..10$ | `run_sweep.sh` `SWEEP_D` default | yes (only $d=8$ currently selected in `DIMS_TO_RUN`) |
| initial condition | $x_0\sim\mathcal N(-1,0.05^2)$, $y_0\sim\mathcal N(0,0.05^2)$ | $\rho_0\propto e^{-((x+1)^2+y^2)/2s_0^2}$, $s_0=0.05$ | yes |
| bias sign/scale | added to drift as $-(\partial_xU+g)$, file value used as-is | $J=\rho\nabla U+\beta^{-1}\nabla\rho+\rho g\,e_x \Rightarrow$ drift $=-(\nabla U+g)$, $g_{\rm ABF}=-\gamma A'$, $g_{\rm ABP}=(\alpha/\beta)\partial_x\ln\rho^1$ (`abfBias`/`abpBias` in `abf.edp`/`abp.edp`, dumped verbatim by `write_frames.idp`) | yes |

No sign flip or rescaling is needed anywhere between the two codes: the
$g(x,t)$ stored on disk is exactly the term `main.jl` subtracts, at the
scale it was written.

Two comments in `main.jl`'s header had drifted from the code and have been
corrected: `kappa=10.0` → `kappa=1.0` (the code has always used
`const KAPPA = 1.0`, matching `edp/`'s `kappa0=1` default), and the launch
example `julia --threads=32 meanfield.jl` → `main.jl` (the file's actual
name).

## 6. Output

One file per `d` in `DIMS_TO_RUN`:
`meanfield_crossing_fractions_d<d>.html` — 3 panels (Unbiased | ABP($\alpha$)
| ABF($\alpha$)); solid = fraction passed 0, dashed = fraction passed 0 then
1, colored by $\alpha$.

## 7. Animation (`gif.jl`)

Trajectory-recording counterpart of `main.jl`, in the style of
`../finite_particles/gif.jl`: same potential, same precomputed-bias-field
loader/interpolator, same reflected integrators, but the full $(x,y)$ path
of every particle is stored instead of only first-passage times, and
rendered as an animated GIF (GR backend, matching
`../finite_particles/gif.jl`'s choice for GIF export — `main.jl` uses
PlotlyJS instead, since it only needs a static interactive HTML figure).

It is zoomed in on the **early transient only**: simulated up to
`SIM_TMAX = 0.5` rather than the bias files' full domain $T=5$ (the bias
files themselves still cover $[0,5]$ — `TFINAL=5` is kept as the constant
used to clamp/interpolate `bias_at`, only the simulated/animated window is
shortened), and densely sampled throughout (119 frames over 1000 steps,
`fps=100`, so the GIF shows the initial pile-up breaking apart rather than
the long tail of already-mixed particles drifting to steady state.

For each $d\in\{2,10\}$ it renders a $2\times4$ grid, potential contour in
the background, particles overlaid as a red scatter:

```
row 1:  ABP(α=0.125) | ABP(α=1) | ABP(α=4) | ABP(α=16)
row 2:  Unbiased     | ABF(α=1) | ABF(α=4) | ABF(α=16)
```

ABP sweeps all four alphas; ABF shares the $\alpha=1,4,16$ triple with ABP
(a matched-$\alpha$ comparison of the two algorithms), with Unbiased
taking the remaining slot in row 2. All 8 panels for a given $d$ reuse one
`Params` (one seed, reseeded identically before each method's run), so
every panel sees the *same* noise realization and initial draw — the only
thing that differs between panels is the bias itself. Requires the bias
files for $(\alpha,d)\in\{0.125,1,4,16\}\times\{2,10\}$ for `abp`, and
$(\alpha,d)\in\{1,4,16\}\times\{2,10\}$ for `abf` — all present in
`biases/`.

Output: `meanfield_particles_d2.gif`, `meanfield_particles_d10.gif`.

## 8. Files and usage

| file | role |
|---|---|
| `main.jl` | potential, bias-field loader/interpolator, the three first-passage-time integrators (unbiased/ABP/ABF), the parallel $\alpha$-sweep, the crossing-fraction figure. |
| `gif.jl` | same model/bias loader, trajectory-recording integrators, the $d\in\{2,10\}$ particle-animation GIFs. |
| `biases/` | precomputed $g(x,t)$ grids from `../edp/`, one file per (algorithm, $\alpha$, $d$). |

```
julia --threads=32 main.jl
julia gif.jl
```

`DIMS_TO_RUN` (top of `main.jl`) selects which of the precomputed dimensions
$d\in\{2,\dots,10\}$ get simulated. Parallelism is a single
`Threads.@threads` loop over the flattened $(\alpha,\text{rep})$ job list
for ABP+ABF plus the unbiased replicates. `gif.jl` is single-threaded.
