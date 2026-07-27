# Toy ABF/ABP Paper Run: Procedure, Metrics, and Main Conclusions

This note summarizes the full `paper` preset run saved in
`classic_md/results/toy_abf_abp_paper/`.

Note:
This file summarizes the saved run already present on disk.
After this summary was written, the code was extended with a four-way
`ABF-bin` / `ABF-kernel` / `ABP-kernel` / `ABP-bin+smooth` comparison, and the
offline ABF paths were corrected so they estimate the bias target `G*` with the
proper `gamma` prefactor.
If the `paper` preset is rerun with the current code, the static, Fourier, and
memory-window ABF numbers will therefore differ from the ones quoted below.

The implementation lives in `classic_md/toy_abf_abp_experiments.py`.

## 1. Model and exact targets

The main toy model uses the periodic reaction coordinate `z in T = [-pi, pi)` and an
orthogonal coordinate `y in R`, with

```text
xi(z, y) = z
A0(z) = Delta / 2 * (1 - cos(2 z))
k_kappa(z) = k0 * exp(kappa * cos z)
V_kappa(z, y) = A0(z) - (1 / (2 beta)) log k_kappa(z) + 0.5 * k_kappa(z) * y^2
```

For this model:

- the free energy along `z` is `A(z) = A0(z) + constant`;
- the conditional law is `Y | Z = z ~ N(0, 1 / (beta * k_kappa(z)))`;
- the local mean force is `E[f_kappa(Z, Y) | Z = z] = A0'(z)`;
- the conditional force variance is
  `Var(f_kappa | z) = kappa^2 * sin^2(z) / (2 beta^2)`.

At tempered equilibrium,

```text
pi_alpha(z) propto exp(-beta * A0(z) / (1 + alpha))
gamma = alpha / (1 + alpha)
G*(z) = gamma * A0'(z)
```

The static experiments target `G*` directly. The online experiments compare the running
bias estimate `G_hat_t` to this same target.

## 2. What the full `paper` run varied

The exact settings are recorded in
`classic_md/results/toy_abf_abp_paper/manifest.json`.

### Static benchmark

- `num_repeats = 320`
- `N in {128, 256, 512, 1024, 2048, 4096, 8192}`
- `h in {0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25}`
- `alpha in {1, 4, 9}`
- `kappa in {0, 0.5, 1, 2}`
- `grid_size = 1024`

### Fourier experiment

- `num_repeats = 384`
- `N = 8192`
- `alpha = 4`
- `kappa = 1`
- `h in {0.02, 0.05, 0.08, 0.18}`
- `grid_size = 1024`

### Online harmonic experiment

- `num_replicates = 24`
- `num_walkers = 96`
- `num_steps = 6000`
- `dt = 4e-4`
- `alpha = 4`
- `h = 0.08`
- `k0 = 20`
- saved every `20` steps

### Memory-window experiment

- same toy model
- `num_replicates = 24`
- `num_walkers = 64`
- `num_steps = 6000`
- `dt = 4e-4`
- `alpha = 4`
- `h = 0.08`
- evaluated at `t = 1.0`
- `T_w in {0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6}`

### Orthogonal-relaxation control

- `kappa in {0.5, 1.5}`
- `k0 in {5, 10, 20, 50}`
- `num_replicates = 20`
- `num_walkers = 96`
- `num_steps = 4500`
- `dt = 4e-4`

### Hidden-barrier control

- replaced the harmonic fiber by
  `U_fib(z, y) = lambda * (y^2 - 1)^2 + c * y * sin(z)`
- `lambda = 8`, `c = 2`
- exact `z`-reference profiles computed by 1D quadrature in `y`
- `num_replicates = 16`
- `num_walkers = 96`
- `num_steps = 7000`
- `dt = 3e-4`

## 3. Procedure used in each experiment

### 3.1 Static benchmark

This is the cleanest statistical comparison because there is no dynamics in the data
generation step.

For each setting:

1. Draw independent `Z_i ~ pi_alpha` by inverse-CDF sampling.
2. Draw independent `Y_i | Z_i = z ~ N(0, 1 / (beta * k_kappa(z)))`.
3. Compute the exact local force label `f_i = f_kappa(Z_i, Y_i)`.
4. Estimate `G*` with both methods from the same `Z_i` sample.

Estimators:

```text
ABF:
  Ghat_F(z) = gamma * average of force labels in the z-bin containing z

ABP:
  phat_h(z) = periodic Gaussian KDE from the Z_i only
  Ghat_P(z) = -(alpha / beta) * d_z log phat_h(z)
```

Important detail: ABF uses force labels, while ABP only uses occupancy information.

### 3.2 Fourier experiment

Use the same static iid sampling protocol, then compute the Fourier coefficients of

```text
e_k = F[Ghat - G*](k)
```

and average `|e_k|^2` across repeated draws.

### 3.3 Online harmonic experiment

This is the interacting-walker experiment. Each replicate simulates `96` walkers with
Euler-Maruyama time stepping on the 2D overdamped dynamics.

ABF:

- keeps a running binwise average of the local force labels seen so far;
- applies `gamma` times that running local-force estimate as the bias.

ABP:

- estimates the current marginal density from the current walker cloud by a periodic KDE;
- differentiates `log phat_t` to build the current bias.

Both methods start with walkers initialized near the same `z`-well.

### 3.4 Memory-window experiment

This experiment is not a second online ABF/ABP simulation. Instead, it builds one common
history under the true target bias `G*`, then asks how well a trailing window of length
`T_w` can reconstruct `G*`.

That isolates the effect of reusing old data.

For each replicate:

1. simulate the toy dynamics under the true bias `G*`;
2. store `(Z_s, f_s)` along the path;
3. at the evaluation time `t = 1.0`, use only samples from `[t - T_w, t]`;
4. build the ABF and ABP estimators from that same trailing window.

### 3.5 Orthogonal-relaxation control

Repeat the online harmonic experiment while varying:

- `kappa`, which changes the conditional force variance;
- `k0`, which changes orthogonal mixing speed without changing the equilibrium force
  variance formula.

This is meant to separate pure label-noise effects from slow orthogonal equilibration.

### 3.6 Hidden-barrier control

Replace the harmonic `y`-fiber by a conditional double well.
The `z`-reference free energy and mean force are then computed numerically by quadrature.

This is a negative control: ABF can have low within-channel variance and still be biased
if the walkers fail to move between orthogonal channels.

## 4. Metrics used

All spatial integrals are computed numerically on a periodic uniform `z`-grid.

### Weighted ISE

Used in the static experiments and also as the online force-error metric:

```text
ISE_weighted(Ghat) = int |Ghat(z) - G*(z)|^2 * pi_alpha(z) dz
```

This emphasizes errors where the tempered target puts mass.

### Unweighted ISE

Also reported in the static experiment:

```text
ISE_unweighted(Ghat) = (1 / (2 pi)) * int |Ghat(z) - G*(z)|^2 dz
```

This is useful because the weighted version can hide errors in low-density regions.

### Variance against bandwidth

For each repeated setting, the code computes the pointwise estimator variance across
replicates and then integrates it against `pi_alpha`:

```text
Var_weighted(Ghat) = int Var(Ghat(z)) * pi_alpha(z) dz
```

This is what is plotted on the log-log bandwidth figures.

### Fourier error spectrum

For each replicate:

```text
e_k = F[Ghat - G*](k)
```

The plotted quantity is:

```text
E |e_k|^2
```

as a function of mode `k`.

### Online force error `E_G(t)`

In the online harmonic experiments,

```text
E_G(t) = int |Ghat_t(z) - G*(z)|^2 * pi_alpha(z) dz
```

This is exactly the weighted force ISE applied to the current time-dependent bias estimate.

### Online KL divergence `H(p_t | pi_alpha)`

Numerically, `p_t` is the current kernel estimate of the `z`-marginal from the walker cloud,
and the code computes:

```text
H(p_t | pi_alpha) = int p_t(z) * log(p_t(z) / pi_alpha(z)) dz
```

This measures how close the current `z`-marginal is to the tempered target distribution.

### Roughness `R_G(t)`

This is the squared `L2` norm of the spatial derivative of the estimated bias:

```text
R_G(t) = int |d_z Ghat_t(z)|^2 dz
```

Numerically, the derivative is computed on the periodic grid with a spectral derivative.

Large `R_G(t)` means the estimated bias is highly oscillatory.

### Well transitions

The online code tracks how often a walker switches between the two `z`-wells, using a
binary well label derived from the sign of `cos(z)`.

This is a simple exploration diagnostic.

### Channel fraction

In the hidden-barrier control, the code also tracks the fraction of walkers with `y > 0`.
That is not a quality metric by itself; it is a diagnostic for whether both orthogonal
channels are actually being explored.

## 5. Main conclusions from the saved `paper` run

### 5.1 The static bandwidth-scaling result is the strongest success

At the representative setting `alpha = 4`, `kappa = 1`, `N = 1024`, `h = 0.08`:

- ABP weighted-variance slope over all tested bandwidths: about `-3.06`
- ABF weighted-variance slope over the moderate-bandwidth range: about `-1.04`

This is the cleanest confirmation of the intended message:

- ABF behaves like averaging noisy force labels;
- ABP behaves like differentiating a noisy density estimate.

If the smallest bandwidth is included, the ABF slope looks steeper (`-1.60`), but that is
not the asymptotic regime. At `h ~ 0.02`, the ABF estimator already has a noticeable
empty-bin effect.

### 5.2 The `kappa` sweep cleanly separates the two methods

At `alpha = 4`, `N = 1024`, `h = 0.08`:

- ABF weighted ISE rises from about `0.719` at `kappa = 0` to about `0.808` at `kappa = 2`
- ABP weighted ISE stays flat at about `28.07` for all `kappa`

This is exactly the expected qualitative separation:

- ABF sees the local-force label noise;
- ABP does not, because it only uses the `Z_i` marginal.

### 5.3 The Fourier plots confirm that ABP amplifies high-frequency noise

For the high-frequency tail (`k >= 50`):

- at `h = 0.02`, ABF mean tail power is about `1.4e-5`
- at `h = 0.02`, ABP mean tail power is about `1.6e-1`

So at small bandwidth, ABP injects dramatically more high-frequency force noise.

At larger bandwidths the ABP tail is strongly damped, which is also expected: stronger KDE
smoothing suppresses the differentiated noise.

### 5.4 The online run shows a sharp tradeoff: ABP redistributes faster, but its force is much rougher

Final online means:

- ABF final force error `E_G(t_end)` about `0.0263`
- ABP final force error `E_G(t_end)` about `161.9`
- ABF final KL about `0.179`
- ABP final KL about `0.0359`
- ABF final roughness about `2.59e3`
- ABP final roughness about `4.80e5`

Interpretation:

- ABP flattens the `z`-marginal more aggressively;
- ABP gets the target marginal faster in KL;
- but the associated force estimate is much noisier and much more oscillatory;
- ABF gives a much cleaner force estimate, but slower marginal redistribution.

So the online result supports the "ABP explores fast but differentiates noise" picture.

### 5.5 The memory-window study supports the "ABF can reuse old data" story

At the tested evaluation time, both methods are best around `T_w = 0.8`, but:

- ABF minimum weighted error is about `0.792`
- ABP minimum weighted error is about `257.2`

So in this implementation, increasing the window helps ABF substantially and never makes it
bad again over the tested range. ABP also improves at first, but remains much noisier.

One caveat: the current run does not show a clean U-shaped ABP curve. It shows "decrease
then flatten." So the window-lag story is only partially visible here.

### 5.6 The `k0` control suggests the main harmonic difference is statistical, not geometric

For ABF, increasing `k0` mildly improves the final force error, especially at larger
`kappa`:

- at `kappa = 1.5`, ABF final force error drops from about `0.0341` at `k0 = 5`
  to about `0.0279` at `k0 = 50`

For ABP, the final force error stays of the same order across both `k0` and `kappa`:

- roughly `150` to `190` in all tested cases

That is consistent with the intended control:

- `kappa` changes the ABF label-noise level;
- `k0` changes orthogonal relaxation speed;
- in this harmonic model, the dominant ABP issue is still the differentiated occupancy noise.

### 5.7 The hidden-barrier control remains a negative control

Final hidden-barrier means:

- ABF final force error about `2.02`
- ABP final force error about `270.9`
- both methods end with channel fraction near `0.995`

That last point matters: both methods stayed almost entirely in one `y`-channel.

So this test confirms the intended negative-control message:

- force averaging does not solve hidden orthogonal metastability;
- if the walkers miss one orthogonal channel, the force estimate can still be badly biased.

ABF is still much better than ABP here, but it is no longer anywhere near the clean
harmonic-model accuracy.

## 6. Was the full run "enough"?

For the central statistical claim, yes.

The static benchmark is the decisive part, and the `paper` run is large enough to support:

- the `h^-1` versus `h^-3` variance contrast;
- the `kappa` sensitivity of ABF and `kappa` insensitivity of ABP;
- the extra high-frequency noise created by differentiating the KDE.

For the online comparison, the run is good enough for a qualitative conclusion, but it is
not the final word on best-tuned ABP versus best-tuned ABF.

Reasons:

- the online experiments use one fixed bandwidth `h = 0.08`;
- the walker count is fixed at `96`;
- the ABP force error is so large that more tuning would be worth trying before making a
  very strong performance claim.

So the clean message is:

- the static experiment convincingly demonstrates the statistical mechanism;
- the online experiment already shows the same mechanism qualitatively;
- but the online comparison could still be refined by retuning `h`, increasing walkers,
  or adding a larger-`N` online sweep.

## 7. Why the fixed-`N` force estimates look noisy

The representative force-estimate panel uses:

- `alpha = 4`
- `kappa = 1`
- `N = 1024`
- `h = 0.08`

At that setting, ABF has about `2 pi / h ~ 79` bins.
So with `N = 1024`, the expected number of samples per bin is only:

- about `13` on average;
- about `6.6` in the lowest-density regions;
- about `21.8` in the highest-density regions.

That is already a noisy local average.

For ABP, the situation is even harsher because the estimator differentiates the KDE:

```text
Var(ABP force estimate) ~ 1 / (N h^3)
```

At `N = 1024` and `h = 0.08`, `N h^3` is small, so differentiation amplifies the local
occupancy noise very strongly.

This is why the fixed-`N` figure is useful:

- it is not showing a bug;
- it is showing the finite-sample mechanism directly.

## 8. Files to inspect first

- `classic_md/results/toy_abf_abp_paper/suite_overview.png`
- `classic_md/results/toy_abf_abp_paper/static/four_panel_summary.png`
- `classic_md/results/toy_abf_abp_paper/static/kappa_sweep.png`
- `classic_md/results/toy_abf_abp_paper/fourier/fourier_spectrum.png`
- `classic_md/results/toy_abf_abp_paper/online/online_convergence.png`
- `classic_md/results/toy_abf_abp_paper/memory_window/memory_window_error.png`
- `classic_md/results/toy_abf_abp_paper/orthogonal_relaxation/orthogonal_relaxation_control.png`
- `classic_md/results/toy_abf_abp_paper/hidden_barrier/hidden_barrier_summary.png`

## 9. Important caveat on the current implementation

The static ABF "best over bandwidth" curve plateaus around `0.70` instead of decaying
to zero over the tested `N` range. That means the current ABF implementation is not purely
variance-limited in that regime.

The most likely causes are:

- the piecewise-constant bin estimator;
- the discrete set of tested bandwidths;
- residual bias from the coarse-bin approximation when the variance becomes small.

That does not affect the main bandwidth-scaling conclusion, but it does mean the static
`N`-rate comparison should be interpreted more cautiously than the variance-versus-bandwidth
and `kappa`-sweep results.
