from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np

try:
    from classic_md.jax_abf_abp import stationary_bias, stationary_gamma, stationary_marginal_density
except ModuleNotFoundError:
    from jax_abf_abp import stationary_bias, stationary_gamma, stationary_marginal_density

LOG_ERROR_FLOOR = 1e-16


@dataclass(frozen=True)
class RunData:
    path: Path
    method: str
    alpha: float
    gamma: float
    theta: float
    beta: float
    burn_in: float
    estimator_label: str
    times: np.ndarray
    x_grid: np.ndarray
    free_energy: np.ndarray
    free_energy_gradient: np.ndarray
    marginal_mean: np.ndarray
    marginal_var: np.ndarray
    mean_force_mean: np.ndarray
    mean_force_var: np.ndarray
    bias_mean: np.ndarray
    bias_var: np.ndarray


@dataclass(frozen=True)
class PlotScales:
    snapshot_y_limits: dict[str, tuple[float, float]]
    error_y_limits: dict[str, tuple[float, float]]
    x_limits: tuple[float, float]
    time_limits: tuple[float, float]


def load_run(path: Path) -> RunData:
    with np.load(path, allow_pickle=False) as data:
        config_raw = data["config_json"].item()
        config = json.loads(config_raw)
        method = str(data["method"].item())
        alpha = float(data["alpha"].item())
        gamma = float(data["gamma"].item())
        theta = float(config["potential"].get("theta", 0.0))
        beta = float(config["potential"]["beta"])
        burn_in = float(config["estimator"]["burn_in"])
        estimator_timescale = config["estimator"].get("estimator_timescale")
        if estimator_timescale is None:
            estimator_label = "estimator=running mean"
        else:
            estimator_label = f"estimator_timescale={float(estimator_timescale):g}"
        return RunData(
            path=path,
            method=method,
            alpha=alpha,
            gamma=gamma,
            theta=theta,
            beta=beta,
            burn_in=burn_in,
            estimator_label=estimator_label,
            times=np.asarray(data["times"]),
            x_grid=np.asarray(data["x_grid"]),
            free_energy=np.asarray(data["free_energy"]),
            free_energy_gradient=np.asarray(data["free_energy_gradient"]),
            marginal_mean=np.asarray(data["marginal_mean"]),
            marginal_var=np.asarray(data["marginal_var"]),
            mean_force_mean=np.asarray(data["mean_force_mean"]),
            mean_force_var=np.asarray(data["mean_force_var"]),
            bias_mean=np.asarray(data["bias_mean"]),
            bias_var=np.asarray(data["bias_var"]),
        )


def reference_targets(run: RunData) -> dict[str, np.ndarray]:
    gamma_eff = stationary_gamma(run.method, run.alpha, run.gamma if run.method.lower() == "abf" else None)
    target_marginal = np.asarray(
        stationary_marginal_density(run.x_grid, run.free_energy, run.beta, gamma_eff)
    )
    target_force = run.free_energy_gradient
    target_bias = np.asarray(
        stationary_bias(run.free_energy_gradient, run.method, run.alpha, run.gamma)
    )
    return {
        "marginal": target_marginal,
        "mean_force": target_force,
        "bias": target_bias,
    }


def pick_snapshot_indices(num_times: int, num_snapshots: int) -> np.ndarray:
    if num_times <= 1:
        return np.array([0], dtype=int)
    return np.unique(np.linspace(0, num_times - 1, min(num_snapshots, num_times), dtype=int))


def relative_l2_error(curves: np.ndarray, target: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    target_norm = np.sqrt(np.trapezoid(target**2, x_grid))
    target_norm = max(target_norm, 1e-12)
    diff = curves - target[None, :]
    return np.sqrt(np.trapezoid(diff**2, x_grid, axis=1)) / target_norm


def relative_l2_std(variances: np.ndarray, target: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    target_norm = np.sqrt(np.trapezoid(target**2, x_grid))
    target_norm = max(target_norm, 1e-12)
    return np.sqrt(np.trapezoid(variances, x_grid, axis=1)) / target_norm


def method_label(run: RunData) -> str:
    return f"{run.method.upper()}  alpha={run.alpha:g}  gamma={run.gamma:g}  theta={run.theta:g}"


def make_axis_limits(values: list[np.ndarray], *, nonnegative: bool = False, symmetric: bool = False) -> tuple[float, float]:
    finite_values = [np.ravel(value[np.isfinite(value)]) for value in values if np.size(value) > 0]
    finite_values = [value for value in finite_values if value.size > 0]
    if not finite_values:
        return (0.0, 1.0)

    merged = np.concatenate(finite_values)
    ymin = float(np.min(merged))
    ymax = float(np.max(merged))

    if symmetric:
        bound = max(abs(ymin), abs(ymax))
        if bound == 0.0:
            bound = 1.0
        padding = 0.05 * bound
        return (-bound - padding, bound + padding)

    if nonnegative:
        ymax = max(ymax, 0.0)
        if ymax == 0.0:
            ymax = 1.0
        return (0.0, 1.05 * ymax)

    if ymax == ymin:
        padding = 0.05 * max(1.0, abs(ymax))
        return (ymin - padding, ymax + padding)

    padding = 0.05 * (ymax - ymin)
    return (ymin - padding, ymax + padding)


def make_log_axis_limits(values: list[np.ndarray]) -> tuple[float, float]:
    finite_values = [np.ravel(value[np.isfinite(value) & (value > 0.0)]) for value in values if np.size(value) > 0]
    finite_values = [value for value in finite_values if value.size > 0]
    if not finite_values:
        return (LOG_ERROR_FLOOR, 1.0)

    merged = np.concatenate(finite_values)
    ymin = max(float(np.min(merged)), LOG_ERROR_FLOOR)
    ymax = max(float(np.max(merged)), ymin)
    if ymax == ymin:
        return (0.5 * ymin, 2.0 * ymax)

    log_min = np.log10(ymin)
    log_max = np.log10(ymax)
    padding = 0.05 * (log_max - log_min)
    return (10.0 ** (log_min - padding), 10.0 ** (log_max + padding))


def compute_plot_scales(
    runs: list[RunData],
    num_snapshots: int,
    sigma_scale: float,
    error_scale: str,
) -> PlotScales:
    snapshot_values: dict[str, list[np.ndarray]] = {
        "marginal": [],
        "mean_force": [],
        "bias": [],
    }
    error_values: dict[str, list[np.ndarray]] = {
        "marginal": [],
        "mean_force": [],
        "bias": [],
    }

    xmins = [float(np.min(run.x_grid)) for run in runs]
    xmaxs = [float(np.max(run.x_grid)) for run in runs]
    tmins = [float(np.min(run.times)) for run in runs]
    tmaxs = [float(np.max(run.times)) for run in runs]

    for run in runs:
        targets = reference_targets(run)
        snapshot_indices = pick_snapshot_indices(run.times.size, num_snapshots)

        for idx in snapshot_indices:
            metric_series = {
                "marginal": (run.marginal_mean[idx], run.marginal_var[idx], targets["marginal"], True),
                "mean_force": (run.mean_force_mean[idx], run.mean_force_var[idx], targets["mean_force"], False),
                "bias": (run.bias_mean[idx], run.bias_var[idx], targets["bias"], False),
            }
            for metric, (mean_curve, var_curve, target_curve, clamp_lower) in metric_series.items():
                std_curve = np.sqrt(np.maximum(var_curve, 0.0))
                lower = mean_curve - sigma_scale * std_curve
                upper = mean_curve + sigma_scale * std_curve
                if clamp_lower:
                    lower = np.maximum(lower, 0.0)
                snapshot_values[metric].extend([target_curve, mean_curve, lower, upper])

        error_values["marginal"].extend(
            [
                relative_l2_error(run.marginal_mean, targets["marginal"], run.x_grid),
                relative_l2_std(run.marginal_var, targets["marginal"], run.x_grid),
            ]
        )
        error_values["mean_force"].extend(
            [
                relative_l2_error(run.mean_force_mean, targets["mean_force"], run.x_grid),
                relative_l2_std(run.mean_force_var, targets["mean_force"], run.x_grid),
            ]
        )
        error_values["bias"].extend(
            [
                relative_l2_error(run.bias_mean, targets["bias"], run.x_grid),
                relative_l2_std(run.bias_var, targets["bias"], run.x_grid),
            ]
        )

    snapshot_y_limits = {
        "marginal": make_axis_limits(snapshot_values["marginal"], nonnegative=True),
        "mean_force": make_axis_limits(snapshot_values["mean_force"], symmetric=True),
        "bias": make_axis_limits(snapshot_values["bias"], symmetric=True),
    }
    if error_scale == "log":
        error_y_limits = {
            metric: make_log_axis_limits(values)
            for metric, values in error_values.items()
        }
    else:
        error_y_limits = {
            metric: make_axis_limits(values, nonnegative=True)
            for metric, values in error_values.items()
        }

    return PlotScales(
        snapshot_y_limits=snapshot_y_limits,
        error_y_limits=error_y_limits,
        x_limits=(min(xmins), max(xmaxs)),
        time_limits=(min(tmins), max(tmaxs)),
    )


def plot_snapshots(
    run: RunData,
    output_dir: Path,
    num_snapshots: int,
    sigma_scale: float,
    scales: PlotScales,
) -> Path:
    targets = reference_targets(run)
    indices = pick_snapshot_indices(run.times.size, num_snapshots)
    fig, axes = plt.subplots(
        nrows=indices.size,
        ncols=3,
        figsize=(14.0, 3.4 * indices.size),
        squeeze=False,
        constrained_layout=True,
    )

    for row, idx in enumerate(indices):
        time = run.times[idx]
        series = [
            ("marginal", "Marginal Density", run.marginal_mean[idx], run.marginal_var[idx], targets["marginal"], "#1f77b4"),
            ("mean_force", "Mean Force", run.mean_force_mean[idx], run.mean_force_var[idx], targets["mean_force"], "#d62728"),
            ("bias", "Bias", run.bias_mean[idx], run.bias_var[idx], targets["bias"], "#2ca02c"),
        ]
        for col, (metric, title, mean_curve, var_curve, target_curve, color) in enumerate(series):
            ax = axes[row, col]
            std_curve = np.sqrt(np.maximum(var_curve, 0.0))
            lower = mean_curve - sigma_scale * std_curve
            upper = mean_curve + sigma_scale * std_curve
            if col == 0:
                lower = np.maximum(lower, 0.0)
            ax.plot(run.x_grid, target_curve, color="black", linestyle="--", linewidth=1.5, label="reference")
            ax.plot(run.x_grid, mean_curve, color=color, linewidth=1.8, label="experiment mean")
            ax.fill_between(run.x_grid, lower, upper, color=color, alpha=0.22, label=f"mean ± {sigma_scale:g} sigma")
            ax.set_title(f"{title} at t={time:.3f}")
            ax.set_xlim(*scales.x_limits)
            ax.set_ylim(*scales.snapshot_y_limits[metric])
            ax.grid(alpha=0.25)
            if row == indices.size - 1:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel(method_label(run))
            if row == 0:
                ax.legend(loc="best", fontsize=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run.path.stem}_snapshots.png"
    fig.suptitle(
        f"{method_label(run)}   burn_in={run.burn_in:g}   {run.estimator_label}",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_error_traces(run: RunData, output_dir: Path, scales: PlotScales, error_scale: str) -> Path:
    targets = reference_targets(run)
    metrics = [
        ("marginal", "Marginal Density", run.marginal_mean, run.marginal_var, targets["marginal"], "#1f77b4"),
        ("mean_force", "Mean Force", run.mean_force_mean, run.mean_force_var, targets["mean_force"], "#d62728"),
        ("bias", "Bias", run.bias_mean, run.bias_var, targets["bias"], "#2ca02c"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.8), squeeze=False, constrained_layout=True)

    for ax, (metric, title, mean_curves, variances, target, color) in zip(axes[0], metrics):
        err = relative_l2_error(mean_curves, target, run.x_grid)
        std = relative_l2_std(variances, target, run.x_grid)
        if error_scale == "log":
            err = np.maximum(err, LOG_ERROR_FLOOR)
            std = np.maximum(std, LOG_ERROR_FLOOR)
            ax.set_yscale("log")
        ax.plot(run.times, err, color=color, linewidth=2.0, label="relative L2 error")
        ax.plot(run.times, std, color=color, linewidth=1.6, linestyle=":", label="relative L2 std")
        if run.burn_in > 0.0:
            ax.axvline(run.burn_in, color="black", linestyle="--", linewidth=1.0, label="burn-in")
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("relative L2")
        ax.set_xlim(*scales.time_limits)
        ax.set_ylim(*scales.error_y_limits[metric])
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run.path.stem}_errors.png"
    fig.suptitle(method_label(run), fontsize=14)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot classical-MD ABF/ABP experiments against reference targets.")
    parser.add_argument("results", nargs="+", type=Path, help="One or more .npz files produced by jax_abf_abp.py")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "plots")
    parser.add_argument("--num-snapshots", type=int, default=4)
    parser.add_argument("--sigma-scale", type=float, default=2.0)
    parser.add_argument("--error-scale", choices=("log", "linear"), default="log")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    runs = [load_run(result_path) for result_path in args.results]
    scales = compute_plot_scales(runs, args.num_snapshots, args.sigma_scale, args.error_scale)

    generated: list[Path] = []
    for run in runs:
        generated.append(plot_snapshots(run, args.output_dir, args.num_snapshots, args.sigma_scale, scales))
        generated.append(plot_error_traces(run, args.output_dir, scales, args.error_scale))

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
