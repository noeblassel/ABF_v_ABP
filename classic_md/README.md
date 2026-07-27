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

## GIF output

`--make-gif` writes a `*.gif` next to the result file by default. The animation shows saved particle positions on top of the rotated potential contours, with optional short trajectory trails.
