# Toy ABF/ABP Four-Way Paper Run: Main Results

This note summarizes the full rerun saved in
`classic_md/results/toy_abf_abp_paper_four_way/`.

It uses the current implementation in `classic_md/toy_abf_abp_experiments.py`, including:

- the four-way static/Fourier comparison
  - `ABF-bin`
  - `ABF-kernel`
  - `ABP-kernel`
  - `ABP-bin+smooth`
- the corrected offline ABF scaling, so the static, Fourier, and memory-window ABF paths
  are compared to the proper target `G*(z) = gamma A0'(z)`.

The key question for this rerun was:

> Is the earlier ABF-versus-ABP gap mostly an artifact of using a bin estimator for ABF
> and a KDE for ABP, or does it persist when the smoothers are matched more closely?

## 1. Short answer

The four-way comparison shows that the gap is **not** mainly a bin-versus-KDE artifact.

What changed:

- `ABF-kernel` is much better than `ABF-bin`.
- `ABP-bin+smooth` is only slightly better than `ABP-kernel`.

What did **not** change:

- the ABP-family estimators are still far noisier than the ABF-family estimators;
- the ABP-family estimators are still essentially insensitive to `kappa`;
- the ABP-family estimators still carry much more high-frequency force error.

So the core difference is still:

- **ABF** estimates a smoothed conditional mean of force labels;
- **ABP** differentiates a smoothed marginal-density estimate.

That algorithmic distinction survives the change of smoother.

## 2. What the four-way run varied

The full preset is recorded in
`classic_md/results/toy_abf_abp_paper_four_way/manifest.json`.

The main static comparison used:

- `320` iid repeats per setting
- `N in {128, 256, 512, 1024, 2048, 4096, 8192}`
- `h in {0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25}`
- `alpha in {1, 4, 9}`
- `kappa in {0, 0.5, 1, 2}`

The Fourier comparison used:

- `384` iid repeats
- `N = 8192`
- `alpha = 4`
- `kappa = 1`
- `h in {0.02, 0.05, 0.08, 0.18}`

The online, memory-window, `k0`, and hidden-barrier sections were also rerun so the full
result tree is consistent, but the new four-way comparison itself applies to the static and
Fourier experiments.

## 3. Main static result: matched smoothing does not erase the ABF/ABP gap

At the representative setting

- `alpha = 4`
- `kappa = 1`
- `N = 1024`
- `h = 0.08`

the weighted force ISEs were:

```text
ABF-bin         0.0418
ABF-kernel      0.0100
ABP-kernel     28.0711
ABP-bin+smooth 25.3462
```

This is the most important table in the rerun.

Interpretation:

- switching ABF from bins to kernel smoothing improves it by about a factor of `4.17`
  at this fixed setting;
- switching ABP from kernel to bins+smooth improves it only by about `10%`;
- both ABP variants remain orders of magnitude worse than both ABF variants for force
  estimation at this finite sample size.

So:

- the **ABF family** is sensitive to the smoother choice;
- the **ABP family** is much less sensitive to the smoother choice;
- the **ABF vs ABP** gap is much larger than the **KDE vs bins+smooth** gap inside ABP.

## 4. Best-over-bandwidth comparison

At `alpha = 4`, `kappa = 1`, after selecting the best tested bandwidth separately for each
estimator, the weighted ISE versus `N` is:

```text
N = 1024
ABF-bin         0.04177
ABF-kernel      0.00841
ABP-kernel      1.10757
ABP-bin+smooth  1.05561
```

```text
N = 8192
ABF-bin         0.00790
ABF-kernel      0.00137
ABP-kernel      0.33057
ABP-bin+smooth  0.34431
```

Main takeaways:

- `ABF-kernel` is the best of the four estimators across the tested `N` range.
- `ABF-bin` is clearly worse than `ABF-kernel`, so the earlier ABF bias plateau was indeed
  partly a property of the bin estimator.
- `ABP-kernel` and `ABP-bin+smooth` are very close to each other, which suggests that the
  main ABP difficulty is not the particular smoother but the differentiation step itself.

This is exactly the clarification the four-way comparison was supposed to provide.

## 5. Bias-variance tradeoff in the four-way comparison

At the same representative setting `alpha = 4`, `kappa = 1`, `N = 1024`, `h = 0.08`,
the weighted bias-variance decomposition is:

```text
ABF-bin:
  weighted bias^2   0.02530
  weighted variance 0.01647

ABF-kernel:
  weighted bias^2   0.00312
  weighted variance 0.00689

ABP-kernel:
  weighted bias^2   0.07683
  weighted variance 27.99431

ABP-bin+smooth:
  weighted bias^2   0.06009
  weighted variance 25.28608
```

Interpretation:

- `ABF-kernel` improves on `ABF-bin` in **both** bias and variance here.
- `ABP-bin+smooth` improves on `ABP-kernel` only modestly, again in both bias and variance.
- The ABP-family error is dominated by **variance**, not bias, at this finite sample size.
- The ABF-family error is much more balanced, especially for `ABF-kernel`.

So the simple statement is:

- the matched-kernel experiment does **not** reveal a hidden “ABF has lower variance but
  catastrophically higher bias” story;
- instead it shows that a better ABF smoother removes much of the previous ABF bias while
  preserving a huge variance advantage over ABP.

## 6. The `kappa` sweep still isolates the algorithmic difference

At fixed `alpha = 4`, `N = 1024`, `h = 0.08`, the weighted ISE across `kappa` is:

```text
ABF-bin:
  0.0278, 0.0315, 0.0418, 0.0832    for kappa = 0, 0.5, 1, 2

ABF-kernel:
  0.00637, 0.00724, 0.01000, 0.02023

ABP-kernel:
  28.0711, 28.0711, 28.0711, 28.0711

ABP-bin+smooth:
  25.3462, 25.3462, 25.3462, 25.3462
```

This is one of the cleanest diagnostics in the whole study.

Interpretation:

- both ABF variants worsen as `kappa` increases, because they use noisy force labels;
- both ABP variants are flat in `kappa`, because they only use the `Z`-marginal;
- the smoother choice inside each family does not change that qualitative separation.

So the `kappa` control confirms that the dominant distinction is still:

- force-label averaging for ABF,
- occupancy differentiation for ABP.

## 7. Fourier result: the differentiated-density estimators still amplify high frequencies

Using the high-frequency tail average over modes `k >= 50`:

At `h = 0.02`:

```text
ABF-bin         9.13e-06
ABF-kernel      5.32e-07
ABP-kernel      1.58e-01
ABP-bin+smooth  1.39e-01
```

At `h = 0.08`:

```text
ABF-bin         2.64e-05
ABF-kernel      4.02e-12
ABP-kernel      2.55e-08
ABP-bin+smooth  1.28e-08
```

Interpretation:

- `ABF-kernel` has the cleanest high-frequency behavior of all four estimators.
- `ABP-kernel` and `ABP-bin+smooth` are again very close.
- The two ABP variants still show the characteristic “differentiate the noise” behavior.

So the Fourier view supports the same conclusion as the static MSE comparison:

- the smoother basis inside ABP is secondary;
- the differentiation mechanism is primary.

## 8. What happened to the bandwidth scaling?

At `alpha = 4`, `kappa = 1`, `N = 1024`, the empirical weighted-variance slopes were:

```text
ABF-bin         about -1.60 over the full tested range
ABF-kernel      about -0.18 over the full tested range
ABP-kernel      about -3.06
ABP-bin+smooth  about -3.08
```

Interpretation:

- the two ABP variants give the expected `h^-3` scaling very clearly;
- `ABF-bin` still has a decreasing-variance trend, but the smallest-bandwidth regime is
  visibly contaminated by bin artifacts and empty-bin effects;
- `ABF-kernel` does **not** show a clean empirical `h^-1` law over this tested range.

That last point is important. The most likely reason is that `ABF-kernel` is a ratio of two
smoothed fields:

- smoothed force density
- smoothed particle density

and over the tested `h` range the variance reduction and ratio effects flatten the curve.

So the four-way rerun clarifies something useful:

- the `h^-1` versus `h^-3` story is very clean for `ABF-bin` versus the ABP estimators;
- once ABF is upgraded to a kernel-regression estimator, the force MSE improves sharply,
  but the simple empirical variance law is less visually clean on the tested grid.

That does **not** weaken the main conclusion. It only means the matched-kernel ABF estimator
is no longer best summarized by a simple slope fit.

## 9. Online and control experiments in the new full run

These are still the original two-way experiments, but they were rerun so the whole result
tree is consistent.

### Online harmonic experiment

Final means:

```text
ABF:
  force error  0.0263
  KL           0.1788
  roughness    2.59e3

ABP:
  force error  1.62e2
  KL           0.0359
  roughness    4.80e5
```

Interpretation:

- ABF still gives a much better force estimate;
- ABP still reaches the target marginal more quickly;
- ABP still pays for that with a much rougher force.

### Memory-window experiment

With the corrected offline ABF scaling, the ABF window study now looks much cleaner:

```text
ABF minimum mean weighted ISE   about 0.0942
ABP minimum mean weighted ISE   about 257.2
```

The ABF curve decreases strongly and then plateaus.
The ABP curve decreases strongly and then flattens, with a very weak late upturn.

### Orthogonal-relaxation control

The same qualitative message survives:

- ABF final force error remains small and mildly improves with larger `k0`;
- ABP force error remains of order `150` to `190` across the tested `k0` and `kappa`.

### Hidden-barrier control

This remains a negative control:

```text
ABF final hidden-barrier force error   about 2.02
ABP final hidden-barrier force error   about 270.9
```

Both methods still stay almost entirely in one `y`-channel, so neither solves hidden
orthogonal metastability automatically.

## 10. Main interpretation of the four-way study

This rerun answers the methodological question cleanly.

### 10.1 What was a smoother artifact?

The earlier `ABF-bin` estimator had noticeable bias from its piecewise-constant structure.
Moving to `ABF-kernel` removes most of that.

So yes:

- part of the earlier ABF bias was a **bin-estimator artifact**.

### 10.2 What was not a smoother artifact?

The large ABF-versus-ABP gap in finite-sample force estimation did **not** disappear when
the smoothers were matched more closely.

In fact:

- `ABF-kernel` is still dramatically better than `ABP-kernel`;
- `ABP-kernel` and `ABP-bin+smooth` are very similar.

So no:

- the core ABF-versus-ABP difference is **not** explained by “ABF used bins, ABP used KDE”.

### 10.3 What does the four-way run say the real difference is?

The dominant difference is still algorithmic:

- **ABF** estimates a conditional mean of force labels;
- **ABP** differentiates a density estimate.

Changing the smoother changes constants and bias levels.
It does not change that structural distinction.

## 11. Practical conclusion

If the goal is **force estimation** in this analytically controlled toy model:

- `ABF-kernel` is the best estimator tested here.

If the goal is to isolate the **core ABF-versus-ABP statistical mechanism**:

- the matched-kernel comparison strengthens the original claim rather than weakening it.

The clean supervisor-facing conclusion is:

> The earlier ABF advantage was not just a bin-versus-KDE artifact.
> After matching the smoothers more closely, the ABP-family estimators remain far noisier
> because they differentiate occupancy fluctuations, whereas ABF estimates a smoothed
> conditional mean of force labels.

## 12. Files to inspect first

- `classic_md/results/toy_abf_abp_paper_four_way/static/four_way_force_estimates.png`
- `classic_md/results/toy_abf_abp_paper_four_way/static/four_way_bias_variance.png`
- `classic_md/results/toy_abf_abp_paper_four_way/static/four_way_mse_vs_sample_size.png`
- `classic_md/results/toy_abf_abp_paper_four_way/static/four_way_variance_vs_bandwidth.png`
- `classic_md/results/toy_abf_abp_paper_four_way/static/four_way_kappa_sweep.png`
- `classic_md/results/toy_abf_abp_paper_four_way/fourier/four_way_fourier_spectrum.png`
- `classic_md/results/toy_abf_abp_paper_four_way/online/online_convergence.png`
- `classic_md/results/toy_abf_abp_paper_four_way/hidden_barrier/hidden_barrier_summary.png`
