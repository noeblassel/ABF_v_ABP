# Classical-MD ABF / ABP in JAX

This folder contains a trajectory-based JAX implementation of the entropic-potential model from the top-level `README.md` and `edp/entropic_potential.idp`.

The implementation is intentionally **not** an interacting particle system:

- each trajectory carries its own running estimator on a uniform grid in the reaction coordinate `x`;
- `burn_in` keeps the dynamics unbiased until the estimator is activated;
- after `burn_in`, the marginal and mean-force estimators are plain running means
  over all kernel observations collected since activation;
- ABF uses the running estimate of the conditional mean force;
- ABP uses the running estimate of the marginal density and its derivative.

The simulator can run many independent trajectories in parallel. This is useful if you want to study the variance of the estimated marginal density or mean force across repeated classical-MD runs.
It also supports a rotated potential, while keeping the estimator CV fixed to the lab-frame coordinate `x`, so that this CV can be deliberately misaligned with the natural channel of the landscape.

## Files

- `jax_abf_abp.py`: entropic potential, free energy, ABF/ABP dynamics, CLI.
- `plot_results.py`: compares saved experiments to the reference stationary marginal, mean force, and bias.
- `toy_abf_abp_experiments.py`: the new periodic toy-model suite for the analytically controlled ABF/ABP comparisons.
- `run_toy_abf_abp_experiments.sh`: small wrapper around the required `ML4MD-py311` interpreter for the toy-model suite.
- `results/`: default output location for `.npz` files created by the CLI.
- `plots/`: default output location for summary figures created by `plot_results.py`.

## Example

```bash
/home/jhimbert/miniconda3/envs/ML4MD-py311/bin/python classic_md/jax_abf_abp.py \
  --method abf \
  --num-trajectories 64 \
  --num-steps 5000 \
  --dt 0.001 \
  --alpha 5.0 \
  --burn-in 0.5 \
  --save-stride 100
```

To rotate the potential by an angle `theta` (in radians) and export a trajectory GIF:

```bash
/home/jhimbert/miniconda3/envs/ML4MD-py311/bin/python classic_md/jax_abf_abp.py \
  --method abp \
  --num-trajectories 24 \
  --num-steps 4000 \
  --dt 0.001 \
  --alpha 5.0 \
  --theta 0.4 \
  --save-stride 20 \
  --make-gif \
  --gif-max-particles 24 \
  --gif-trail-length 10
```

For ABF you may override the bias amplitude directly with `--gamma`; otherwise the code uses
`gamma = alpha / (1 + alpha)` and also supports `--alpha inf`, which gives `gamma = 1`.
For ABP, `gamma` is not used: the stationary bias amplitude is fixed by `alpha`.

The saved `.npz` file contains:

- `times`, `x_grid`
- `free_energy`, `free_energy_gradient`
  - for `theta = 0`, these are the closed-form reference profiles along `x`
  - for `theta != 0`, they are numerical lab-frame references along the estimator CV `x`
- mean and variance over trajectories of the running marginal estimator
- mean and variance over trajectories of the running mean-force estimator
- mean and variance over trajectories of the bias field
- instantaneous one-step kernel diagnostics

If you pass `--store-particles` and/or `--store-estimators`, the file also includes the raw saved trajectory states and per-trajectory estimator fields.
If you pass `--make-gif`, the script captures particle snapshots internally even if you do not store them in the `.npz` file.

## Visualization

You can compare any saved run to the reference targets with:

```bash
/home/jhimbert/miniconda3/envs/ML4MD-py311/bin/python classic_md/plot_results.py \
  classic_md/results/smoke_abf.npz \
  classic_md/results/smoke_abp.npz
```

For each input file the script writes two figures in `classic_md/plots/`:

- `*_snapshots.png`: selected times showing experiment mean, `mean ± 2 sigma`, and the reference target for the marginal density, mean force, and bias.
- `*_errors.png`: relative `L2` error and relative `L2` standard deviation versus time for those same quantities. By default the error axis is logarithmic; pass `--error-scale linear` to recover a linear scale.

## Toy-model suite

`toy_abf_abp_experiments.py` implements the analytically controlled periodic toy model

- `V_kappa(z, y) = A_0(z) - (2 beta)^{-1} log k_kappa(z) + 0.5 k_kappa(z) y^2`
- `A_0(z) = Delta / 2 * (1 - cos(2 z))`
- `k_kappa(z) = k_0 exp(kappa cos z)`

and writes a self-contained result tree under `classic_md/results/<exp_name>/`.

The suite covers:

- the static equilibrium benchmark with weighted and unweighted ISEs;
- a four-way static estimator comparison:
  - `ABF-bin`
  - `ABF-kernel`
  - `ABP-kernel`
  - `ABP-bin+smooth`
- the Fourier-spectrum comparison;
- the online interacting-walker experiment;
- the memory-window comparison;
- the orthogonal-relaxation (`k_0`) control;
- the hidden-barrier stress test.

The simplest way to run it is:

```bash
sh classic_md/run_toy_abf_abp_experiments.sh \
  --profile smoke \
  --exp-name toy_abf_abp_smoke
```

The available profiles are:

- `smoke`: fast sanity check that still produces all figures.
- `standard`: larger sweeps suitable for routine comparison runs.
- `paper`: the heaviest preset, close to the sweep sizes described in the experiment notes.

You can also run just a subset:

```bash
sh classic_md/run_toy_abf_abp_experiments.sh \
  --profile standard \
  --experiments static fourier online \
  --exp-name toy_abf_abp_main
```

Each experiment folder stores its own `.npz` data and figures next to one another, for example:

- `classic_md/results/<exp_name>/static/four_panel_summary.png`
- `classic_md/results/<exp_name>/static/four_way_force_estimates.png`
- `classic_md/results/<exp_name>/static/four_way_variance_vs_bandwidth.png`
- `classic_md/results/<exp_name>/static/four_way_mse_vs_sample_size.png`
- `classic_md/results/<exp_name>/static/four_way_bias_variance.png`
- `classic_md/results/<exp_name>/static/four_way_kappa_sweep.png`
- `classic_md/results/<exp_name>/fourier/fourier_spectrum.png`
- `classic_md/results/<exp_name>/fourier/four_way_fourier_spectrum.png`
- `classic_md/results/<exp_name>/online/online_convergence.png`
- `classic_md/results/<exp_name>/memory_window/memory_window_error.png`
- `classic_md/results/<exp_name>/orthogonal_relaxation/orthogonal_relaxation_control.png`
- `classic_md/results/<exp_name>/hidden_barrier/hidden_barrier_summary.png`

If `static` and `online` are both run, the root of the experiment directory also gets a supervisor-style overview figure:

- `classic_md/results/<exp_name>/suite_overview.png`

## GIF output

`--make-gif` writes a `*.gif` next to the result file by default. The animation shows saved particle positions on top of the rotated potential contours, with optional short trajectory trails.
