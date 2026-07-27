from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

TWO_PI = 2.0 * math.pi
PLOT_DPI = 180
METHODS = ("abf", "abp")
FOUR_WAY_ESTIMATORS = ("abf_bin", "abf_kernel", "abp_kernel", "abp_bin_smooth")
METHOD_COLORS = {
    "abf": "#2166ac",
    "abp": "#b2182b",
}
METHOD_LABELS = {
    "abf": "ABF",
    "abp": "ABP",
}
FOUR_WAY_COLORS = {
    "abf_bin": "#2166ac",
    "abf_kernel": "#67a9cf",
    "abp_kernel": "#b2182b",
    "abp_bin_smooth": "#ef8a62",
}
FOUR_WAY_LABELS = {
    "abf_bin": "ABF bin",
    "abf_kernel": "ABF kernel",
    "abp_kernel": "ABP kernel",
    "abp_bin_smooth": "ABP bin+smooth",
}
FOUR_WAY_LINESTYLES = {
    "abf_bin": "-",
    "abf_kernel": "--",
    "abp_kernel": "-",
    "abp_bin_smooth": "--",
}


@dataclass(frozen=True)
class ToyModelConfig:
    beta: float = 1.0
    delta: float = 6.0
    k0: float = 20.0
    kappa: float = 1.0


@dataclass(frozen=True)
class StaticExperimentConfig:
    num_repeats: int
    batch_size: int
    sample_sizes: tuple[int, ...]
    bandwidths: tuple[float, ...]
    alphas: tuple[float, ...]
    kappas: tuple[float, ...]
    grid_size: int
    force_overlay_repeats: int
    representative_alpha: float
    representative_kappa: float
    representative_sample_size: int
    representative_bandwidth: float


@dataclass(frozen=True)
class FourierExperimentConfig:
    num_repeats: int
    batch_size: int
    sample_size: int
    alpha: float
    kappa: float
    bandwidths: tuple[float, ...]
    grid_size: int


@dataclass(frozen=True)
class OnlineExperimentConfig:
    num_replicates: int
    num_walkers: int
    num_steps: int
    dt: float
    alpha: float
    bandwidth: float
    grid_size: int
    save_stride: int
    init_z_center: float
    init_z_std: float
    init_y_std: float


@dataclass(frozen=True)
class MemoryWindowExperimentConfig:
    num_replicates: int
    num_walkers: int
    num_steps: int
    dt: float
    alpha: float
    bandwidth: float
    grid_size: int
    history_stride: int
    evaluation_time: float
    window_sizes: tuple[float, ...]
    init_z_center: float
    init_z_std: float
    init_y_std: float


@dataclass(frozen=True)
class K0ControlExperimentConfig:
    num_replicates: int
    num_walkers: int
    num_steps: int
    dt: float
    alpha: float
    bandwidth: float
    grid_size: int
    save_stride: int
    kappas: tuple[float, ...]
    k0_values: tuple[float, ...]
    init_z_center: float
    init_z_std: float
    init_y_std: float


@dataclass(frozen=True)
class HiddenBarrierExperimentConfig:
    num_replicates: int
    num_walkers: int
    num_steps: int
    dt: float
    alpha: float
    bandwidth: float
    grid_size: int
    save_stride: int
    beta: float
    delta: float
    fiber_lambda: float
    coupling: float
    y_max: float
    y_grid_size: int
    init_z_center: float
    init_z_std: float
    init_y_center: float
    init_y_std: float


@dataclass(frozen=True)
class SuiteProfile:
    model: ToyModelConfig
    static: StaticExperimentConfig
    fourier: FourierExperimentConfig
    online: OnlineExperimentConfig
    memory: MemoryWindowExperimentConfig
    k0_control: K0ControlExperimentConfig
    hidden_barrier: HiddenBarrierExperimentConfig


def wrap_periodic(z: np.ndarray) -> np.ndarray:
    return ((z + math.pi) % TWO_PI) - math.pi


def periodic_grid(grid_size: int) -> np.ndarray:
    return np.linspace(-math.pi, math.pi, grid_size, endpoint=False)


def periodic_dz(grid_size: int) -> float:
    return TWO_PI / float(grid_size)


def spectral_modes(grid_size: int) -> np.ndarray:
    return np.fft.fftfreq(grid_size, d=1.0 / grid_size)


def spectral_derivative(values: np.ndarray, modes: np.ndarray) -> np.ndarray:
    transformed = np.fft.fft(values, axis=-1)
    derivative = np.fft.ifft(transformed * (1j * modes), axis=-1).real
    return derivative


def periodic_integral(values: np.ndarray, dz: float) -> np.ndarray:
    return dz * np.sum(values, axis=-1)


def gamma_from_alpha(alpha: float) -> float:
    return alpha / (1.0 + alpha)


def a0(z: np.ndarray, delta: float) -> np.ndarray:
    return 0.5 * delta * (1.0 - np.cos(2.0 * z))


def a0_prime(z: np.ndarray, delta: float) -> np.ndarray:
    return delta * np.sin(2.0 * z)


def stiffness(z: np.ndarray, kappa: float, k0: float) -> np.ndarray:
    return k0 * np.exp(kappa * np.cos(z))


def stiffness_log_derivative(z: np.ndarray, kappa: float) -> np.ndarray:
    return -kappa * np.sin(z)


def stiffness_derivative(z: np.ndarray, kappa: float, k0: float) -> np.ndarray:
    return stiffness(z, kappa, k0) * stiffness_log_derivative(z, kappa)


def harmonic_local_force(
    z: np.ndarray,
    y: np.ndarray,
    *,
    beta: float,
    delta: float,
    kappa: float,
    k0: float,
) -> np.ndarray:
    return (
        a0_prime(z, delta)
        - 0.5 * stiffness_log_derivative(z, kappa) / beta
        + 0.5 * stiffness_derivative(z, kappa, k0) * (y**2)
    )


def harmonic_y_drift(z: np.ndarray, y: np.ndarray, *, kappa: float, k0: float) -> np.ndarray:
    return -stiffness(z, kappa, k0) * y


def harmonic_tempered_density(
    z_grid: np.ndarray,
    *,
    beta: float,
    delta: float,
    alpha: float,
) -> np.ndarray:
    weights = np.exp(-beta * a0(z_grid, delta) / (1.0 + alpha))
    dz = periodic_dz(z_grid.size)
    return weights / periodic_integral(weights, dz)


def harmonic_true_bias(
    z_grid: np.ndarray,
    *,
    alpha: float,
    delta: float,
) -> np.ndarray:
    return gamma_from_alpha(alpha) * a0_prime(z_grid, delta)


def harmonic_force_variance(
    z_grid: np.ndarray,
    *,
    beta: float,
    kappa: float,
) -> np.ndarray:
    return 0.5 * (kappa**2) * (np.sin(z_grid) ** 2) / (beta**2)


def make_tempered_sampler(
    *,
    alpha: float,
    beta: float,
    delta: float,
    cdf_grid_size: int = 65_536,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(-math.pi, math.pi, cdf_grid_size + 1)
    midpoints = 0.5 * (grid[:-1] + grid[1:])
    weights = np.exp(-beta * a0(midpoints, delta) / (1.0 + alpha))
    cumulative = np.concatenate(([0.0], np.cumsum(weights)))
    cumulative /= cumulative[-1]
    return cumulative, grid


def sample_tempered_z(
    rng: np.random.Generator,
    sampler: tuple[np.ndarray, np.ndarray],
    shape: tuple[int, ...],
) -> np.ndarray:
    cumulative, grid = sampler
    uniforms = rng.random(shape)
    samples = np.interp(uniforms, cumulative, grid)
    return wrap_periodic(samples)


def sample_harmonic_y(
    rng: np.random.Generator,
    z: np.ndarray,
    *,
    beta: float,
    kappa: float,
    k0: float,
) -> np.ndarray:
    std = np.sqrt(1.0 / (beta * stiffness(z, kappa, k0)))
    return rng.normal(scale=std, size=z.shape)


def periodic_bin_count(bandwidth: float) -> tuple[int, float]:
    num_bins = max(4, int(round(TWO_PI / bandwidth)))
    actual_bandwidth = TWO_PI / float(num_bins)
    return num_bins, actual_bandwidth


def periodic_bin_indices(z: np.ndarray, num_bins: int) -> np.ndarray:
    scaled = (wrap_periodic(z) + math.pi) * num_bins / TWO_PI
    return np.floor(scaled).astype(np.int64) % num_bins


def batched_bincount(
    indices: np.ndarray,
    num_bins: int,
    values: np.ndarray | None = None,
) -> np.ndarray:
    batch_size = indices.shape[0]
    accumulator = np.zeros((batch_size, num_bins), dtype=np.float64)
    batch_ids = np.broadcast_to(np.arange(batch_size)[:, None], indices.shape)
    increments = 1.0 if values is None else values
    np.add.at(accumulator, (batch_ids, indices), increments)
    return accumulator


def cic_histogram_periodic(
    samples: np.ndarray,
    grid_size: int,
    values: np.ndarray | None = None,
) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[None, :]
    if values is not None:
        values = np.asarray(values)
        if values.ndim == 1:
            values = values[None, :]
    scaled = (wrap_periodic(samples) + math.pi) * grid_size / TWO_PI
    left = np.floor(scaled).astype(np.int64) % grid_size
    frac = scaled - np.floor(scaled)
    right = (left + 1) % grid_size
    histogram = np.zeros((samples.shape[0], grid_size), dtype=np.float64)
    batch_ids = np.broadcast_to(np.arange(samples.shape[0])[:, None], samples.shape)
    left_values = 1.0 - frac
    right_values = frac
    if values is not None:
        left_values = left_values * values
        right_values = right_values * values
    np.add.at(histogram, (batch_ids, left), left_values)
    np.add.at(histogram, (batch_ids, right), right_values)
    return histogram


def smooth_periodic_field_and_derivative(
    field: np.ndarray,
    *,
    bandwidth: float,
    modes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    multiplier = np.exp(-0.5 * (bandwidth * modes) ** 2)
    transformed = np.fft.fft(field, axis=-1)
    smoothed = np.fft.ifft(transformed * multiplier[None, :], axis=-1).real
    smoothed_derivative = np.fft.ifft(
        transformed * multiplier[None, :] * (1j * modes)[None, :],
        axis=-1,
    ).real
    return smoothed, smoothed_derivative


def smooth_periodic_field(
    field: np.ndarray,
    *,
    bandwidth: float,
    modes: np.ndarray,
) -> np.ndarray:
    smoothed, _ = smooth_periodic_field_and_derivative(field, bandwidth=bandwidth, modes=modes)
    return smoothed


def smooth_density_from_histogram(
    histogram: np.ndarray,
    *,
    bandwidth: float,
    dz: float,
    modes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts = histogram.sum(axis=-1, keepdims=True)
    empirical_density = histogram / np.maximum(counts, 1.0) / dz
    return smooth_periodic_field_and_derivative(empirical_density, bandwidth=bandwidth, modes=modes)


def density_from_samples(
    samples: np.ndarray,
    *,
    bandwidth: float,
    grid_size: int,
    dz: float,
    modes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    histogram = cic_histogram_periodic(samples, grid_size)
    return smooth_density_from_histogram(histogram, bandwidth=bandwidth, dz=dz, modes=modes)


def abp_force_from_samples(
    samples: np.ndarray,
    *,
    alpha: float,
    beta: float,
    bandwidth: float,
    grid_size: int,
    dz: float,
    modes: np.ndarray,
    floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    density, density_derivative = density_from_samples(
        samples,
        bandwidth=bandwidth,
        grid_size=grid_size,
        dz=dz,
        modes=modes,
    )
    safe_density = np.maximum(density, floor)
    force = -(alpha / beta) * density_derivative / safe_density
    return force, safe_density


def abf_force_from_samples(
    z_samples: np.ndarray,
    force_samples: np.ndarray,
    *,
    gamma: float,
    bandwidth: float,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    num_bins, actual_bandwidth = periodic_bin_count(bandwidth)
    sample_bins = periodic_bin_indices(z_samples, num_bins)
    counts = batched_bincount(sample_bins, num_bins)
    sums = batched_bincount(sample_bins, num_bins, values=force_samples)
    mean_force = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0.0)
    grid_bins = periodic_bin_indices(grid[None, :], num_bins)[0]
    estimate = gamma * mean_force[:, grid_bins]
    return estimate, counts, actual_bandwidth


def abf_kernel_force_from_histograms(
    density_histogram: np.ndarray,
    force_histogram: np.ndarray,
    *,
    gamma: float,
    bandwidth: float,
    dz: float,
    modes: np.ndarray,
    floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    counts = density_histogram.sum(axis=-1, keepdims=True)
    empirical_density = density_histogram / np.maximum(counts, 1.0) / dz
    empirical_force_density = force_histogram / np.maximum(counts, 1.0) / dz
    smoothed_density = smooth_periodic_field(empirical_density, bandwidth=bandwidth, modes=modes)
    smoothed_force_density = smooth_periodic_field(empirical_force_density, bandwidth=bandwidth, modes=modes)
    estimate = gamma * smoothed_force_density / np.maximum(smoothed_density, floor)
    return estimate, smoothed_density


def piecewise_density_from_counts(
    counts: np.ndarray,
    *,
    sample_size: int,
    bin_width: float,
    grid_lookup: np.ndarray,
) -> np.ndarray:
    return counts[:, grid_lookup] / float(sample_size) / bin_width


def abp_binned_smooth_force_from_counts(
    counts: np.ndarray,
    *,
    sample_size: int,
    alpha: float,
    beta: float,
    bin_width: float,
    grid_lookup: np.ndarray,
    modes: np.ndarray,
    floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    piecewise_density = piecewise_density_from_counts(
        counts,
        sample_size=sample_size,
        bin_width=bin_width,
        grid_lookup=grid_lookup,
    )
    smoothed_density, smoothed_density_derivative = smooth_periodic_field_and_derivative(
        piecewise_density,
        bandwidth=bin_width,
        modes=modes,
    )
    estimate = -(alpha / beta) * smoothed_density_derivative / np.maximum(smoothed_density, floor)
    return estimate, smoothed_density


def interpolate_periodic_grid(values: np.ndarray, z: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        values = values[None, :]
        squeeze = True
    else:
        squeeze = False
    grid_size = values.shape[-1]
    scaled = (wrap_periodic(z) + math.pi) * grid_size / TWO_PI
    left = np.floor(scaled).astype(np.int64) % grid_size
    frac = scaled - np.floor(scaled)
    right = (left + 1) % grid_size
    batch_ids = np.broadcast_to(np.arange(values.shape[0])[:, None], z.shape)
    interpolated = (1.0 - frac) * values[batch_ids, left] + frac * values[batch_ids, right]
    if squeeze:
        return interpolated[0]
    return interpolated


def weighted_ise(
    estimate: np.ndarray,
    truth: np.ndarray,
    density: np.ndarray,
    dz: float,
) -> np.ndarray:
    error = estimate - truth[None, :]
    return periodic_integral(error**2 * density[None, :], dz)


def unweighted_ise(
    estimate: np.ndarray,
    truth: np.ndarray,
    dz: float,
) -> np.ndarray:
    error = estimate - truth[None, :]
    return periodic_integral(error**2, dz) / TWO_PI


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def save_figure(path: Path, fig: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_with_band(ax: plt.Axes, x: np.ndarray, mean: np.ndarray, std: np.ndarray, *, label: str, color: str) -> None:
    ax.plot(x, mean, color=color, linewidth=2.0, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0.0)


def representative_indices(
    config: StaticExperimentConfig,
) -> tuple[int, int, int, int]:
    alpha_idx = config.alphas.index(config.representative_alpha)
    kappa_idx = config.kappas.index(config.representative_kappa)
    sample_idx = config.sample_sizes.index(config.representative_sample_size)
    bandwidth_idx = config.bandwidths.index(config.representative_bandwidth)
    return alpha_idx, kappa_idx, sample_idx, bandwidth_idx


def run_static_experiment(
    model: ToyModelConfig,
    config: StaticExperimentConfig,
    output_dir: Path,
    seed: int,
) -> dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    grid = periodic_grid(config.grid_size)
    dz = periodic_dz(config.grid_size)
    modes = spectral_modes(config.grid_size)
    num_alphas = len(config.alphas)
    num_kappas = len(config.kappas)
    num_samples = len(config.sample_sizes)
    num_bandwidths = len(config.bandwidths)
    num_methods = len(METHODS)
    num_comparison_estimators = len(FOUR_WAY_ESTIMATORS)
    legacy_abf_idx = METHODS.index("abf")
    legacy_abp_idx = METHODS.index("abp")
    comparison_index = {name: idx for idx, name in enumerate(FOUR_WAY_ESTIMATORS)}

    weighted_ise_values = np.zeros(
        (num_methods, num_alphas, num_kappas, num_samples, num_bandwidths, config.num_repeats),
        dtype=np.float64,
    )
    unweighted_ise_values = np.zeros_like(weighted_ise_values)
    mean_estimate = np.zeros(
        (num_methods, num_alphas, num_kappas, num_samples, num_bandwidths, config.grid_size),
        dtype=np.float64,
    )
    second_moment = np.zeros_like(mean_estimate)
    empty_bin_fraction = np.zeros(
        (num_alphas, num_kappas, num_samples, num_bandwidths, config.num_repeats),
        dtype=np.float64,
    )
    actual_abf_bandwidths = np.zeros(num_bandwidths, dtype=np.float64)
    comparison_weighted_ise_values = np.zeros(
        (num_comparison_estimators, num_alphas, num_kappas, num_samples, num_bandwidths, config.num_repeats),
        dtype=np.float64,
    )
    comparison_unweighted_ise_values = np.zeros_like(comparison_weighted_ise_values)
    comparison_mean_estimate = np.zeros(
        (num_comparison_estimators, num_alphas, num_kappas, num_samples, num_bandwidths, config.grid_size),
        dtype=np.float64,
    )
    comparison_second_moment = np.zeros_like(comparison_mean_estimate)
    comparison_empty_bin_fraction = np.full(
        (
            num_comparison_estimators,
            num_alphas,
            num_kappas,
            num_samples,
            num_bandwidths,
            config.num_repeats,
        ),
        np.nan,
        dtype=np.float64,
    )
    comparison_actual_bandwidths = np.broadcast_to(
        np.asarray(config.bandwidths, dtype=np.float64)[None, :],
        (num_comparison_estimators, num_bandwidths),
    ).copy()

    alpha_targets = {
        alpha: harmonic_tempered_density(
            grid,
            beta=model.beta,
            delta=model.delta,
            alpha=alpha,
        )
        for alpha in config.alphas
    }
    alpha_truths = {
        alpha: harmonic_true_bias(
            grid,
            alpha=alpha,
            delta=model.delta,
        )
        for alpha in config.alphas
    }
    samplers = {
        alpha: make_tempered_sampler(alpha=alpha, beta=model.beta, delta=model.delta)
        for alpha in config.alphas
    }

    overlay_alpha_idx, overlay_kappa_idx, overlay_sample_idx, overlay_bandwidth_idx = representative_indices(config)
    overlay_truth = alpha_truths[config.representative_alpha]
    overlay_density = alpha_targets[config.representative_alpha]
    overlay_estimates = {name: [] for name in FOUR_WAY_ESTIMATORS}

    for alpha_idx, alpha in enumerate(config.alphas):
        target_density = alpha_targets[alpha]
        target_force = alpha_truths[alpha]
        sampler = samplers[alpha]
        gamma = gamma_from_alpha(alpha)
        for sample_idx, sample_size in enumerate(config.sample_sizes):
            for repeat_start in range(0, config.num_repeats, config.batch_size):
                repeat_stop = min(config.num_repeats, repeat_start + config.batch_size)
                batch_size = repeat_stop - repeat_start
                z_samples = sample_tempered_z(rng, sampler, (batch_size, sample_size))
                z_histogram = cic_histogram_periodic(z_samples, config.grid_size)
                sample_bins_by_bandwidth: list[np.ndarray] = []
                counts_by_bandwidth: list[np.ndarray] = []
                grid_lookup_by_bandwidth: list[np.ndarray] = []
                kernel_density_by_bandwidth: list[np.ndarray] = []
                for bandwidth_idx, bandwidth in enumerate(config.bandwidths):
                    abp_density, abp_density_derivative = smooth_density_from_histogram(
                        z_histogram,
                        bandwidth=bandwidth,
                        dz=dz,
                        modes=modes,
                    )
                    abp_force = -(alpha / model.beta) * abp_density_derivative / np.maximum(abp_density, 1e-12)
                    kernel_density_by_bandwidth.append(abp_density)
                    weighted_ise_values[
                        legacy_abp_idx,
                        alpha_idx,
                        :,
                        sample_idx,
                        bandwidth_idx,
                        repeat_start:repeat_stop,
                    ] = weighted_ise(abp_force, target_force, target_density, dz)[None, :]
                    unweighted_ise_values[
                        legacy_abp_idx,
                        alpha_idx,
                        :,
                        sample_idx,
                        bandwidth_idx,
                        repeat_start:repeat_stop,
                    ] = unweighted_ise(abp_force, target_force, dz)[None, :]
                    mean_estimate[legacy_abp_idx, alpha_idx, :, sample_idx, bandwidth_idx] += abp_force.sum(axis=0)
                    second_moment[legacy_abp_idx, alpha_idx, :, sample_idx, bandwidth_idx] += np.sum(
                        abp_force**2,
                        axis=0,
                    )
                    comparison_weighted_ise_values[
                        comparison_index["abp_kernel"],
                        alpha_idx,
                        :,
                        sample_idx,
                        bandwidth_idx,
                        repeat_start:repeat_stop,
                    ] = weighted_ise(abp_force, target_force, target_density, dz)[None, :]
                    comparison_unweighted_ise_values[
                        comparison_index["abp_kernel"],
                        alpha_idx,
                        :,
                        sample_idx,
                        bandwidth_idx,
                        repeat_start:repeat_stop,
                    ] = unweighted_ise(abp_force, target_force, dz)[None, :]
                    comparison_mean_estimate[
                        comparison_index["abp_kernel"],
                        alpha_idx,
                        :,
                        sample_idx,
                        bandwidth_idx,
                    ] += abp_force.sum(axis=0)[None, :]
                    comparison_second_moment[
                        comparison_index["abp_kernel"],
                        alpha_idx,
                        :,
                        sample_idx,
                        bandwidth_idx,
                    ] += np.sum(abp_force**2, axis=0)[None, :]
                    if (
                        alpha_idx == overlay_alpha_idx
                        and sample_idx == overlay_sample_idx
                        and bandwidth_idx == overlay_bandwidth_idx
                        and len(overlay_estimates["abp_kernel"]) < config.force_overlay_repeats
                    ):
                        missing = config.force_overlay_repeats - len(overlay_estimates["abp_kernel"])
                        overlay_estimates["abp_kernel"].extend(abp_force[:missing])

                    num_bins, actual_bandwidth = periodic_bin_count(bandwidth)
                    sample_bins = periodic_bin_indices(z_samples, num_bins)
                    counts = batched_bincount(sample_bins, num_bins)
                    grid_lookup = periodic_bin_indices(grid[None, :], num_bins)[0]
                    sample_bins_by_bandwidth.append(sample_bins)
                    counts_by_bandwidth.append(counts)
                    grid_lookup_by_bandwidth.append(grid_lookup)
                    actual_abf_bandwidths[bandwidth_idx] = actual_bandwidth
                    comparison_actual_bandwidths[comparison_index["abf_bin"], bandwidth_idx] = actual_bandwidth
                    comparison_actual_bandwidths[comparison_index["abp_bin_smooth"], bandwidth_idx] = actual_bandwidth
                    abp_binned_force, _ = abp_binned_smooth_force_from_counts(
                        counts,
                        sample_size=sample_size,
                        alpha=alpha,
                        beta=model.beta,
                        bin_width=actual_bandwidth,
                        grid_lookup=grid_lookup,
                        modes=modes,
                    )
                    comparison_weighted_ise_values[
                        comparison_index["abp_bin_smooth"],
                        alpha_idx,
                        :,
                        sample_idx,
                        bandwidth_idx,
                        repeat_start:repeat_stop,
                    ] = weighted_ise(abp_binned_force, target_force, target_density, dz)[None, :]
                    comparison_unweighted_ise_values[
                        comparison_index["abp_bin_smooth"],
                        alpha_idx,
                        :,
                        sample_idx,
                        bandwidth_idx,
                        repeat_start:repeat_stop,
                    ] = unweighted_ise(abp_binned_force, target_force, dz)[None, :]
                    comparison_mean_estimate[
                        comparison_index["abp_bin_smooth"],
                        alpha_idx,
                        :,
                        sample_idx,
                        bandwidth_idx,
                    ] += abp_binned_force.sum(axis=0)[None, :]
                    comparison_second_moment[
                        comparison_index["abp_bin_smooth"],
                        alpha_idx,
                        :,
                        sample_idx,
                        bandwidth_idx,
                    ] += np.sum(abp_binned_force**2, axis=0)[None, :]
                    comparison_empty_bin_fraction[
                        comparison_index["abp_bin_smooth"],
                        alpha_idx,
                        :,
                        sample_idx,
                        bandwidth_idx,
                        repeat_start:repeat_stop,
                    ] = np.mean(counts == 0.0, axis=1)[None, :]
                    if (
                        alpha_idx == overlay_alpha_idx
                        and sample_idx == overlay_sample_idx
                        and bandwidth_idx == overlay_bandwidth_idx
                        and len(overlay_estimates["abp_bin_smooth"]) < config.force_overlay_repeats
                    ):
                        missing = config.force_overlay_repeats - len(overlay_estimates["abp_bin_smooth"])
                        overlay_estimates["abp_bin_smooth"].extend(abp_binned_force[:missing])

                for kappa_idx, kappa in enumerate(config.kappas):
                    y_samples = sample_harmonic_y(
                        rng,
                        z_samples,
                        beta=model.beta,
                        kappa=kappa,
                        k0=model.k0,
                    )
                    force_samples = harmonic_local_force(
                        z_samples,
                        y_samples,
                        beta=model.beta,
                        delta=model.delta,
                        kappa=kappa,
                        k0=model.k0,
                    )
                    force_histogram = cic_histogram_periodic(z_samples, config.grid_size, values=force_samples)
                    empirical_force_density = force_histogram / float(sample_size) / dz
                    for bandwidth_idx, bandwidth in enumerate(config.bandwidths):
                        sample_bins = sample_bins_by_bandwidth[bandwidth_idx]
                        counts = counts_by_bandwidth[bandwidth_idx]
                        grid_lookup = grid_lookup_by_bandwidth[bandwidth_idx]
                        actual_bandwidth = actual_abf_bandwidths[bandwidth_idx]
                        sums = batched_bincount(sample_bins, counts.shape[1], values=force_samples)
                        mean_force = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0.0)
                        abf_force = gamma * mean_force[:, grid_lookup]
                        weighted_ise_values[
                            legacy_abf_idx,
                            alpha_idx,
                            kappa_idx,
                            sample_idx,
                            bandwidth_idx,
                            repeat_start:repeat_stop,
                        ] = weighted_ise(abf_force, target_force, target_density, dz)
                        unweighted_ise_values[
                            legacy_abf_idx,
                            alpha_idx,
                            kappa_idx,
                            sample_idx,
                            bandwidth_idx,
                            repeat_start:repeat_stop,
                        ] = unweighted_ise(abf_force, target_force, dz)
                        mean_estimate[legacy_abf_idx, alpha_idx, kappa_idx, sample_idx, bandwidth_idx] += abf_force.sum(
                            axis=0
                        )
                        second_moment[legacy_abf_idx, alpha_idx, kappa_idx, sample_idx, bandwidth_idx] += np.sum(
                            abf_force**2,
                            axis=0,
                        )
                        empty_bin_fraction[
                            alpha_idx,
                            kappa_idx,
                            sample_idx,
                            bandwidth_idx,
                            repeat_start:repeat_stop,
                        ] = np.mean(counts == 0.0, axis=1)
                        comparison_weighted_ise_values[
                            comparison_index["abf_bin"],
                            alpha_idx,
                            kappa_idx,
                            sample_idx,
                            bandwidth_idx,
                            repeat_start:repeat_stop,
                        ] = weighted_ise(abf_force, target_force, target_density, dz)
                        comparison_unweighted_ise_values[
                            comparison_index["abf_bin"],
                            alpha_idx,
                            kappa_idx,
                            sample_idx,
                            bandwidth_idx,
                            repeat_start:repeat_stop,
                        ] = unweighted_ise(abf_force, target_force, dz)
                        comparison_mean_estimate[
                            comparison_index["abf_bin"],
                            alpha_idx,
                            kappa_idx,
                            sample_idx,
                            bandwidth_idx,
                        ] += abf_force.sum(axis=0)
                        comparison_second_moment[
                            comparison_index["abf_bin"],
                            alpha_idx,
                            kappa_idx,
                            sample_idx,
                            bandwidth_idx,
                        ] += np.sum(abf_force**2, axis=0)
                        comparison_empty_bin_fraction[
                            comparison_index["abf_bin"],
                            alpha_idx,
                            kappa_idx,
                            sample_idx,
                            bandwidth_idx,
                            repeat_start:repeat_stop,
                        ] = np.mean(counts == 0.0, axis=1)
                        if (
                            alpha_idx == overlay_alpha_idx
                            and kappa_idx == overlay_kappa_idx
                            and sample_idx == overlay_sample_idx
                            and bandwidth_idx == overlay_bandwidth_idx
                            and len(overlay_estimates["abf_bin"]) < config.force_overlay_repeats
                        ):
                            missing = config.force_overlay_repeats - len(overlay_estimates["abf_bin"])
                            overlay_estimates["abf_bin"].extend(abf_force[:missing])

                        smoothed_force_density = smooth_periodic_field(
                            empirical_force_density,
                            bandwidth=bandwidth,
                            modes=modes,
                        )
                        abf_kernel_force = gamma * smoothed_force_density / np.maximum(
                            kernel_density_by_bandwidth[bandwidth_idx],
                            1e-12,
                        )
                        comparison_weighted_ise_values[
                            comparison_index["abf_kernel"],
                            alpha_idx,
                            kappa_idx,
                            sample_idx,
                            bandwidth_idx,
                            repeat_start:repeat_stop,
                        ] = weighted_ise(abf_kernel_force, target_force, target_density, dz)
                        comparison_unweighted_ise_values[
                            comparison_index["abf_kernel"],
                            alpha_idx,
                            kappa_idx,
                            sample_idx,
                            bandwidth_idx,
                            repeat_start:repeat_stop,
                        ] = unweighted_ise(abf_kernel_force, target_force, dz)
                        comparison_mean_estimate[
                            comparison_index["abf_kernel"],
                            alpha_idx,
                            kappa_idx,
                            sample_idx,
                            bandwidth_idx,
                        ] += abf_kernel_force.sum(axis=0)
                        comparison_second_moment[
                            comparison_index["abf_kernel"],
                            alpha_idx,
                            kappa_idx,
                            sample_idx,
                            bandwidth_idx,
                        ] += np.sum(abf_kernel_force**2, axis=0)
                        if (
                            alpha_idx == overlay_alpha_idx
                            and kappa_idx == overlay_kappa_idx
                            and sample_idx == overlay_sample_idx
                            and bandwidth_idx == overlay_bandwidth_idx
                            and len(overlay_estimates["abf_kernel"]) < config.force_overlay_repeats
                        ):
                            missing = config.force_overlay_repeats - len(overlay_estimates["abf_kernel"])
                            overlay_estimates["abf_kernel"].extend(abf_kernel_force[:missing])

    mean_estimate /= float(config.num_repeats)
    second_moment /= float(config.num_repeats)
    variance_estimate = np.maximum(second_moment - mean_estimate**2, 0.0)
    comparison_mean_estimate /= float(config.num_repeats)
    comparison_second_moment /= float(config.num_repeats)
    comparison_variance_estimate = np.maximum(comparison_second_moment - comparison_mean_estimate**2, 0.0)

    target_density_array = np.stack([alpha_targets[alpha] for alpha in config.alphas], axis=0)
    weighted_variance = np.zeros((num_methods, num_alphas, num_kappas, num_samples, num_bandwidths), dtype=np.float64)
    comparison_weighted_variance = np.zeros(
        (num_comparison_estimators, num_alphas, num_kappas, num_samples, num_bandwidths),
        dtype=np.float64,
    )
    for alpha_idx in range(num_alphas):
        weighted_variance[:, alpha_idx] = periodic_integral(
            variance_estimate[:, alpha_idx] * target_density_array[alpha_idx][None, None, None, None, :],
            dz,
        )
        comparison_weighted_variance[:, alpha_idx] = periodic_integral(
            comparison_variance_estimate[:, alpha_idx]
            * target_density_array[alpha_idx][None, None, None, None, :],
            dz,
        )

    payload = {
        "grid": grid,
        "dz": np.asarray(dz),
        "alpha_values": np.asarray(config.alphas),
        "kappa_values": np.asarray(config.kappas),
        "sample_sizes": np.asarray(config.sample_sizes),
        "bandwidths": np.asarray(config.bandwidths),
        "actual_abf_bandwidths": actual_abf_bandwidths,
        "weighted_ise": weighted_ise_values,
        "unweighted_ise": unweighted_ise_values,
        "mean_estimate": mean_estimate,
        "variance_estimate": variance_estimate,
        "weighted_variance": weighted_variance,
        "empty_bin_fraction": empty_bin_fraction,
        "target_density": target_density_array,
        "target_force": np.stack([alpha_truths[alpha] for alpha in config.alphas], axis=0),
        "overlay_truth": overlay_truth,
        "overlay_density": overlay_density,
        "overlay_abf": np.stack(overlay_estimates["abf_bin"], axis=0),
        "overlay_abp": np.stack(overlay_estimates["abp_kernel"], axis=0),
        "comparison_estimator_names": np.asarray(FOUR_WAY_ESTIMATORS),
        "comparison_weighted_ise": comparison_weighted_ise_values,
        "comparison_unweighted_ise": comparison_unweighted_ise_values,
        "comparison_mean_estimate": comparison_mean_estimate,
        "comparison_variance_estimate": comparison_variance_estimate,
        "comparison_weighted_variance": comparison_weighted_variance,
        "comparison_actual_bandwidths": comparison_actual_bandwidths,
        "comparison_empty_bin_fraction": comparison_empty_bin_fraction,
        "comparison_overlay_estimates": np.stack(
            [np.stack(overlay_estimates[name], axis=0) for name in FOUR_WAY_ESTIMATORS],
            axis=0,
        ),
        "theoretical_force_variance": np.stack(
            [harmonic_force_variance(grid, beta=model.beta, kappa=kappa) for kappa in config.kappas],
            axis=0,
        ),
        "config_json": np.asarray(json.dumps(asdict(config), sort_keys=True)),
        "model_json": np.asarray(json.dumps(asdict(model), sort_keys=True)),
    }
    np.savez_compressed(output_dir / "static_data.npz", **payload)

    representative_actual_abf_bandwidth = actual_abf_bandwidths[overlay_bandwidth_idx]
    summary = {
        "representative_alpha": config.representative_alpha,
        "representative_kappa": config.representative_kappa,
        "representative_sample_size": config.representative_sample_size,
        "representative_bandwidth": config.representative_bandwidth,
        "representative_actual_abf_bandwidth": representative_actual_abf_bandwidth,
    }
    save_json(output_dir / "summary.json", summary)
    make_static_plots(model, config, payload, output_dir)
    make_four_way_static_plots(model, config, payload, output_dir)
    return payload


def make_static_plots(
    model: ToyModelConfig,
    config: StaticExperimentConfig,
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    grid = np.asarray(data["grid"])
    dz = float(np.asarray(data["dz"]).item())
    alpha_values = np.asarray(data["alpha_values"])
    kappa_values = np.asarray(data["kappa_values"])
    sample_sizes = np.asarray(data["sample_sizes"])
    bandwidths = np.asarray(data["bandwidths"])
    actual_abf_bandwidths = np.asarray(data["actual_abf_bandwidths"])
    weighted_ise_values = np.asarray(data["weighted_ise"])
    unweighted_ise_values = np.asarray(data["unweighted_ise"])
    mean_estimate = np.asarray(data["mean_estimate"])
    variance_estimate = np.asarray(data["variance_estimate"])
    weighted_variance = np.asarray(data["weighted_variance"])
    target_density = np.asarray(data["target_density"])
    target_force = np.asarray(data["target_force"])
    overlay_truth = np.asarray(data["overlay_truth"])
    overlay_density = np.asarray(data["overlay_density"])
    overlay_abf = np.asarray(data["overlay_abf"])
    overlay_abp = np.asarray(data["overlay_abp"])
    theoretical_force_variance = np.asarray(data["theoretical_force_variance"])

    alpha_idx, kappa_idx, sample_idx, bandwidth_idx = representative_indices(config)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(grid, overlay_truth, color="black", linewidth=2.3, label=r"$G^\star$")
    for estimate in overlay_abf:
        ax.plot(grid, estimate, color=METHOD_COLORS["abf"], alpha=0.28, linewidth=1.0)
    for estimate in overlay_abp:
        ax.plot(grid, estimate, color=METHOD_COLORS["abp"], alpha=0.28, linewidth=1.0)
    ax.set_title(
        f"Representative Force Estimates\nalpha={config.representative_alpha:g}, "
        f"kappa={config.representative_kappa:g}, N={config.representative_sample_size}, "
        f"h={config.representative_bandwidth:g}"
    )
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\widehat G(z)$")
    ax.legend(
        [
            Line2D([], [], color="black", linewidth=2.3),
            Line2D([], [], color=METHOD_COLORS["abf"], linewidth=2.0),
            Line2D([], [], color=METHOD_COLORS["abp"], linewidth=2.0),
        ],
        [r"$G^\star$", "ABF replicates", "ABP replicates"],
        loc="upper right",
    )

    ax = axes[0, 1]
    representative_weighted_variance = weighted_variance[:, alpha_idx, kappa_idx, sample_idx]
    abf_slope = loglog_slope(actual_abf_bandwidths, representative_weighted_variance[METHODS.index("abf")])
    abp_slope = loglog_slope(bandwidths, representative_weighted_variance[METHODS.index("abp")])
    ax.loglog(
        actual_abf_bandwidths,
        representative_weighted_variance[METHODS.index("abf")],
        marker="o",
        color=METHOD_COLORS["abf"],
        linewidth=2.0,
        label=f"ABF slope {abf_slope:.2f}",
    )
    ax.loglog(
        bandwidths,
        representative_weighted_variance[METHODS.index("abp")],
        marker="s",
        color=METHOD_COLORS["abp"],
        linewidth=2.0,
        label=f"ABP slope {abp_slope:.2f}",
    )
    ax.set_title("Variance Against Bandwidth")
    ax.set_xlabel("Bandwidth h")
    ax.set_ylabel(r"$\int \mathrm{Var}(\widehat G)\,\pi_\alpha\,dz$")
    ax.legend(loc="lower left")

    ax = axes[1, 0]
    best_weighted_ise = weighted_ise_values.mean(axis=-1).min(axis=-1)
    for method_idx, method in enumerate(METHODS):
        ax.loglog(
            sample_sizes,
            best_weighted_ise[method_idx, alpha_idx, kappa_idx],
            marker="o",
            linewidth=2.0,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set_title("Best Weighted ISE Against Sample Size")
    ax.set_xlabel("Sample size N")
    ax.set_ylabel("Best weighted ISE")
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    ax.plot(grid, overlay_density, color="#444444", linewidth=2.2, label=rf"$\pi_{{\alpha}}$, alpha={config.representative_alpha:g}")
    ax.plot(
        grid,
        theoretical_force_variance[kappa_idx],
        color="#4d9221",
        linewidth=2.0,
        label=rf"$\mathrm{{Var}}(f\mid z)$, kappa={config.representative_kappa:g}",
    )
    ax.set_title("Analytic Controls")
    ax.set_xlabel("z")
    ax.set_ylabel("Density / variance")
    ax.legend(loc="upper right")
    save_figure(output_dir / "four_panel_summary.png", fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8), constrained_layout=True)
    small_bandwidth_idx = 0
    large_bandwidth_idx = min(len(bandwidths) - 1, 2)
    bandwidth_selection = [small_bandwidth_idx, large_bandwidth_idx]
    for col_idx, selected_bandwidth_idx in enumerate(bandwidth_selection):
        selected_bandwidth = bandwidths[selected_bandwidth_idx]
        for row_idx, method in enumerate(METHODS):
            method_idx = METHODS.index(method)
            ax = axes[row_idx, col_idx]
            mean_curve = mean_estimate[method_idx, alpha_idx, kappa_idx, sample_idx, selected_bandwidth_idx]
            var_curve = variance_estimate[method_idx, alpha_idx, kappa_idx, sample_idx, selected_bandwidth_idx]
            bias_sq = (mean_curve - target_force[alpha_idx]) ** 2
            ax.plot(grid, bias_sq, color="#303030", linewidth=2.0, label=r"Bias$^2$")
            ax.plot(grid, var_curve, color=METHOD_COLORS[method], linewidth=2.0, label="Variance")
            ax.set_yscale("log")
            ax.set_ylim(1e-6, max(np.max(bias_sq), np.max(var_curve), 1e-5) * 1.35)
            ax.set_title(f"{METHOD_LABELS[method]} at h={selected_bandwidth:g}")
            ax.set_xlabel("z")
            ax.set_ylabel("Pointwise contribution")
            ax.legend(loc="upper right")
    save_figure(output_dir / "bias_variance_decomposition.png", fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 5.0), constrained_layout=True)
    fixed_sample_idx = sample_idx
    fixed_bandwidth_idx = bandwidth_idx
    for i, method in enumerate(METHODS):
        method_idx = METHODS.index(method)
        mean_error = weighted_ise_values[method_idx, alpha_idx, :, fixed_sample_idx, fixed_bandwidth_idx].mean(axis=-1)
        std_error = weighted_ise_values[method_idx, alpha_idx, :, fixed_sample_idx, fixed_bandwidth_idx].std(axis=-1)
        plot_with_band(
            axes[i],
            kappa_values,
            mean_error,
            std_error,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
        axes[i].set_title(
            f"{METHOD_LABELS[method]}: Kappa Sweep at N={sample_sizes[fixed_sample_idx]}, h={bandwidths[fixed_bandwidth_idx]:g}, alpha={alpha_values[alpha_idx]:g}"
        )
        axes[i].set_xlabel("kappa")
        axes[i].set_ylabel("Weighted ISE")
        axes[i].legend(loc="upper left")
    save_figure(output_dir / "kappa_sweep.png", fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
    mean_weighted_ise = weighted_ise_values.mean(axis=-1)
    mean_unweighted_ise = unweighted_ise_values.mean(axis=-1)
    for ax, metric, title in (
        (axes[0], mean_weighted_ise, "Weighted ISE"),
        (axes[1], mean_unweighted_ise, "Unweighted ISE"),
    ):
        for method in METHODS:
            method_idx = METHODS.index(method)
            best_bandwidth_idx = np.argmin(metric[method_idx, alpha_idx, kappa_idx], axis=-1)
            best_values = metric[method_idx, alpha_idx, kappa_idx, np.arange(sample_sizes.size), best_bandwidth_idx]
            ax.loglog(
                sample_sizes,
                best_values,
                marker="o",
                linewidth=2.0,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_title(title)
        ax.set_xlabel("Sample size N")
        ax.set_ylabel(title)
        ax.legend(loc="upper right")
    save_figure(output_dir / "mse_vs_sample_size.png", fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    mean_empty_fraction = np.asarray(data["empty_bin_fraction"])[alpha_idx, kappa_idx, sample_idx].mean(axis=-1)
    ax.plot(actual_abf_bandwidths, mean_empty_fraction, marker="o", color=METHOD_COLORS["abf"], linewidth=2.0)
    ax.set_title("ABF Empty-Bin Fraction")
    ax.set_xlabel("Actual ABF bin width")
    ax.set_ylabel("Mean fraction of empty bins")
    save_figure(output_dir / "abf_empty_bin_fraction.png", fig)


def make_four_way_static_plots(
    model: ToyModelConfig,
    config: StaticExperimentConfig,
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    del model
    grid = np.asarray(data["grid"])
    alpha_values = np.asarray(data["alpha_values"])
    kappa_values = np.asarray(data["kappa_values"])
    sample_sizes = np.asarray(data["sample_sizes"])
    target_force = np.asarray(data["target_force"])
    bandwidths = np.asarray(data["bandwidths"])
    comparison_names = [str(name) for name in np.asarray(data["comparison_estimator_names"])]
    comparison_weighted_ise = np.asarray(data["comparison_weighted_ise"])
    comparison_unweighted_ise = np.asarray(data["comparison_unweighted_ise"])
    comparison_mean_estimate = np.asarray(data["comparison_mean_estimate"])
    comparison_variance_estimate = np.asarray(data["comparison_variance_estimate"])
    comparison_weighted_variance = np.asarray(data["comparison_weighted_variance"])
    comparison_actual_bandwidths = np.asarray(data["comparison_actual_bandwidths"])
    comparison_overlay_estimates = np.asarray(data["comparison_overlay_estimates"])
    overlay_truth = np.asarray(data["overlay_truth"])

    alpha_idx, kappa_idx, sample_idx, bandwidth_idx = representative_indices(config)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8), constrained_layout=True, sharex=True, sharey=True)
    for estimator_idx, estimator_name in enumerate(comparison_names):
        ax = axes.flat[estimator_idx]
        ax.plot(grid, overlay_truth, color="black", linewidth=2.2, label=r"$G^\star$")
        for estimate in comparison_overlay_estimates[estimator_idx]:
            ax.plot(
                grid,
                estimate,
                color=FOUR_WAY_COLORS[estimator_name],
                alpha=0.30,
                linewidth=1.0,
            )
        ax.set_title(FOUR_WAY_LABELS[estimator_name])
        ax.set_xlabel("z")
        ax.set_ylabel(r"$\widehat G(z)$")
    handles = [Line2D([], [], color="black", linewidth=2.2)]
    labels = [r"$G^\star$"]
    handles.extend(
        Line2D([], [], color=FOUR_WAY_COLORS[name], linewidth=2.0, linestyle=FOUR_WAY_LINESTYLES[name])
        for name in comparison_names
    )
    labels.extend(f"{FOUR_WAY_LABELS[name]} replicates" for name in comparison_names)
    fig.legend(handles, labels, loc="upper center", ncols=3)
    save_figure(output_dir / "four_way_force_estimates.png", fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    representative_weighted_variance = comparison_weighted_variance[:, alpha_idx, kappa_idx, sample_idx]
    for estimator_idx, estimator_name in enumerate(comparison_names):
        x = comparison_actual_bandwidths[estimator_idx]
        y = representative_weighted_variance[estimator_idx]
        slope = loglog_slope(x, y)
        ax.loglog(
            x,
            y,
            marker="o",
            linewidth=2.0,
            linestyle=FOUR_WAY_LINESTYLES[estimator_name],
            color=FOUR_WAY_COLORS[estimator_name],
            label=f"{FOUR_WAY_LABELS[estimator_name]} slope {slope:.2f}",
        )
    ax.set_title("Four-Way Variance Against Bandwidth")
    ax.set_xlabel("Bandwidth h")
    ax.set_ylabel(r"$\int \mathrm{Var}(\widehat G)\,\pi_\alpha\,dz$")
    ax.legend(loc="lower left")
    save_figure(output_dir / "four_way_variance_vs_bandwidth.png", fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    mean_weighted_ise = comparison_weighted_ise.mean(axis=-1)
    mean_unweighted_ise = comparison_unweighted_ise.mean(axis=-1)
    for ax, metric, title in (
        (axes[0], mean_weighted_ise, "Weighted ISE"),
        (axes[1], mean_unweighted_ise, "Unweighted ISE"),
    ):
        for estimator_idx, estimator_name in enumerate(comparison_names):
            best_bandwidth_idx = np.argmin(metric[estimator_idx, alpha_idx, kappa_idx], axis=-1)
            best_values = metric[
                estimator_idx,
                alpha_idx,
                kappa_idx,
                np.arange(sample_sizes.size),
                best_bandwidth_idx,
            ]
            ax.loglog(
                sample_sizes,
                best_values,
                marker="o",
                linewidth=2.0,
                linestyle=FOUR_WAY_LINESTYLES[estimator_name],
                color=FOUR_WAY_COLORS[estimator_name],
                label=FOUR_WAY_LABELS[estimator_name],
            )
        ax.set_title(f"Four-Way {title}")
        ax.set_xlabel("Sample size N")
        ax.set_ylabel(title)
        ax.legend(loc="upper right")
    save_figure(output_dir / "four_way_mse_vs_sample_size.png", fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8), constrained_layout=True, sharex=True)
    for estimator_idx, estimator_name in enumerate(comparison_names):
        ax = axes.flat[estimator_idx]
        mean_curve = comparison_mean_estimate[
            estimator_idx,
            alpha_idx,
            kappa_idx,
            sample_idx,
            bandwidth_idx,
        ]
        var_curve = comparison_variance_estimate[
            estimator_idx,
            alpha_idx,
            kappa_idx,
            sample_idx,
            bandwidth_idx,
        ]
        bias_sq = (mean_curve - target_force[alpha_idx]) ** 2
        ax.plot(grid, bias_sq, color="#303030", linewidth=2.0, label=r"Bias$^2$")
        ax.plot(
            grid,
            var_curve,
            color=FOUR_WAY_COLORS[estimator_name],
            linewidth=2.0,
            linestyle=FOUR_WAY_LINESTYLES[estimator_name],
            label="Variance",
        )
        ax.set_yscale("log")
        ax.set_ylim(1e-7, max(np.max(bias_sq), np.max(var_curve), 1e-6) * 1.4)
        ax.set_title(
            f"{FOUR_WAY_LABELS[estimator_name]}\nalpha={alpha_values[alpha_idx]:g}, "
            f"kappa={kappa_values[kappa_idx]:g}, N={sample_sizes[sample_idx]}, h={bandwidths[bandwidth_idx]:g}"
        )
        ax.set_xlabel("z")
        ax.set_ylabel("Pointwise contribution")
        ax.legend(loc="upper right")
    save_figure(output_dir / "four_way_bias_variance.png", fig)

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.2), constrained_layout=True)
    for estimator_idx, estimator_name in enumerate(comparison_names):
        mean_error = comparison_weighted_ise[
            estimator_idx,
            alpha_idx,
            :,
            sample_idx,
            bandwidth_idx,
        ].mean(axis=-1)
        std_error = comparison_weighted_ise[
            estimator_idx,
            alpha_idx,
            :,
            sample_idx,
            bandwidth_idx,
        ].std(axis=-1)
        ax = axes.flat[estimator_idx]
        plot_with_band(
            ax,
            kappa_values,
            mean_error,
            std_error,
            label=FOUR_WAY_LABELS[estimator_name],
            color=FOUR_WAY_COLORS[estimator_name],
        )
        ax.set_title(
            f"{FOUR_WAY_LABELS[estimator_name]} at alpha={alpha_values[alpha_idx]:g}, "
            f"N={sample_sizes[sample_idx]}, h={bandwidths[bandwidth_idx]:g}"
        )
        ax.set_xlabel("kappa")
        ax.set_ylabel("Weighted ISE")
        ax.legend(loc="upper left")
    save_figure(output_dir / "four_way_kappa_sweep.png", fig)


def run_fourier_experiment(
    model: ToyModelConfig,
    config: FourierExperimentConfig,
    output_dir: Path,
    seed: int,
) -> dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    grid = periodic_grid(config.grid_size)
    dz = periodic_dz(config.grid_size)
    modes = spectral_modes(config.grid_size)
    target_force = harmonic_true_bias(grid, alpha=config.alpha, delta=model.delta)
    sampler = make_tempered_sampler(alpha=config.alpha, beta=model.beta, delta=model.delta)
    positive_modes = np.arange(config.grid_size // 2 + 1)
    power = np.zeros((len(METHODS), len(config.bandwidths), positive_modes.size), dtype=np.float64)
    comparison_power = np.zeros((len(FOUR_WAY_ESTIMATORS), len(config.bandwidths), positive_modes.size), dtype=np.float64)
    gamma = gamma_from_alpha(config.alpha)

    for repeat_start in range(0, config.num_repeats, config.batch_size):
        repeat_stop = min(config.num_repeats, repeat_start + config.batch_size)
        batch_size = repeat_stop - repeat_start
        z_samples = sample_tempered_z(rng, sampler, (batch_size, config.sample_size))
        z_histogram = cic_histogram_periodic(z_samples, config.grid_size)
        sample_bins_by_bandwidth: list[np.ndarray] = []
        counts_by_bandwidth: list[np.ndarray] = []
        grid_lookup_by_bandwidth: list[np.ndarray] = []
        kernel_density_by_bandwidth: list[np.ndarray] = []
        abp_force_by_bandwidth: list[np.ndarray] = []
        abp_binned_force_by_bandwidth: list[np.ndarray] = []
        for bandwidth_idx, bandwidth in enumerate(config.bandwidths):
            abp_density, abp_density_derivative = smooth_density_from_histogram(
                z_histogram,
                bandwidth=bandwidth,
                dz=dz,
                modes=modes,
            )
            abp_force = -(config.alpha / model.beta) * abp_density_derivative / np.maximum(abp_density, 1e-12)
            abp_force_by_bandwidth.append(abp_force)
            kernel_density_by_bandwidth.append(abp_density)
            num_bins, actual_bandwidth = periodic_bin_count(bandwidth)
            sample_bins = periodic_bin_indices(z_samples, num_bins)
            counts = batched_bincount(sample_bins, num_bins)
            grid_lookup = periodic_bin_indices(grid[None, :], num_bins)[0]
            sample_bins_by_bandwidth.append(sample_bins)
            counts_by_bandwidth.append(counts)
            grid_lookup_by_bandwidth.append(grid_lookup)
            abp_binned_force, _ = abp_binned_smooth_force_from_counts(
                counts,
                sample_size=config.sample_size,
                alpha=config.alpha,
                beta=model.beta,
                bin_width=actual_bandwidth,
                grid_lookup=grid_lookup,
                modes=modes,
            )
            abp_binned_force_by_bandwidth.append(abp_binned_force)
        y_samples = sample_harmonic_y(
            rng,
            z_samples,
            beta=model.beta,
            kappa=config.kappa,
            k0=model.k0,
        )
        force_samples = harmonic_local_force(
            z_samples,
            y_samples,
            beta=model.beta,
            delta=model.delta,
            kappa=config.kappa,
            k0=model.k0,
        )
        force_histogram = cic_histogram_periodic(z_samples, config.grid_size, values=force_samples)
        empirical_force_density = force_histogram / float(config.sample_size) / dz
        for bandwidth_idx, bandwidth in enumerate(config.bandwidths):
            sample_bins = sample_bins_by_bandwidth[bandwidth_idx]
            counts = counts_by_bandwidth[bandwidth_idx]
            grid_lookup = grid_lookup_by_bandwidth[bandwidth_idx]
            sums = batched_bincount(sample_bins, counts.shape[1], values=force_samples)
            mean_force = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0.0)
            abf_force = gamma * mean_force[:, grid_lookup]
            smoothed_force_density = smooth_periodic_field(
                empirical_force_density,
                bandwidth=bandwidth,
                modes=modes,
            )
            abf_kernel_force = gamma * smoothed_force_density / np.maximum(
                kernel_density_by_bandwidth[bandwidth_idx],
                1e-12,
            )
            abp_force = abp_force_by_bandwidth[bandwidth_idx]
            abp_binned_force = abp_binned_force_by_bandwidth[bandwidth_idx]
            for method, estimates in (("abf", abf_force), ("abp", abp_force)):
                spectrum = np.fft.rfft(estimates - target_force[None, :], axis=-1) / config.grid_size
                power[METHODS.index(method), bandwidth_idx] += np.sum(np.abs(spectrum) ** 2, axis=0)
            for method, estimates in (
                ("abf_bin", abf_force),
                ("abf_kernel", abf_kernel_force),
                ("abp_kernel", abp_force),
                ("abp_bin_smooth", abp_binned_force),
            ):
                spectrum = np.fft.rfft(estimates - target_force[None, :], axis=-1) / config.grid_size
                comparison_power[FOUR_WAY_ESTIMATORS.index(method), bandwidth_idx] += np.sum(
                    np.abs(spectrum) ** 2,
                    axis=0,
                )
    power /= float(config.num_repeats)
    comparison_power /= float(config.num_repeats)

    payload = {
        "modes": positive_modes,
        "bandwidths": np.asarray(config.bandwidths),
        "power": power,
        "comparison_estimator_names": np.asarray(FOUR_WAY_ESTIMATORS),
        "comparison_power": comparison_power,
        "target_force": target_force,
        "grid": grid,
        "config_json": np.asarray(json.dumps(asdict(config), sort_keys=True)),
        "model_json": np.asarray(json.dumps(asdict(model), sort_keys=True)),
    }
    np.savez_compressed(output_dir / "fourier_data.npz", **payload)
    make_fourier_plot(config, payload, output_dir)
    make_four_way_fourier_plot(config, payload, output_dir)
    return payload


def make_fourier_plot(
    config: FourierExperimentConfig,
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    modes = np.asarray(data["modes"])
    bandwidths = np.asarray(data["bandwidths"])
    power = np.asarray(data["power"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True, sharey=True)
    for ax, method in zip(axes, METHODS):
        method_idx = METHODS.index(method)
        for bandwidth_idx, bandwidth in enumerate(bandwidths):
            ax.loglog(
                modes[1:],
                power[method_idx, bandwidth_idx, 1:],
                linewidth=2.0,
                label=f"h={bandwidth:g}",
            )
        ax.set_title(f"{METHOD_LABELS[method]} Fourier Error Spectrum")
        ax.set_xlabel("Mode k")
        ax.set_ylabel(r"$\mathbb{E}|e_k|^2$")
        ax.legend(loc="lower left")
    save_figure(output_dir / "fourier_spectrum.png", fig)


def make_four_way_fourier_plot(
    config: FourierExperimentConfig,
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    del config
    modes = np.asarray(data["modes"])
    bandwidths = np.asarray(data["bandwidths"])
    estimator_names = [str(name) for name in np.asarray(data["comparison_estimator_names"])]
    comparison_power = np.asarray(data["comparison_power"])

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8), constrained_layout=True, sharex=True, sharey=True)
    for estimator_idx, estimator_name in enumerate(estimator_names):
        ax = axes.flat[estimator_idx]
        for bandwidth_idx, bandwidth in enumerate(bandwidths):
            ax.loglog(
                modes[1:],
                comparison_power[estimator_idx, bandwidth_idx, 1:],
                linewidth=2.0,
                label=f"h={bandwidth:g}",
            )
        ax.set_title(f"{FOUR_WAY_LABELS[estimator_name]} Fourier Error")
        ax.set_xlabel("Mode k")
        ax.set_ylabel(r"$\mathbb{E}|e_k|^2$")
        ax.legend(loc="lower left")
    save_figure(output_dir / "four_way_fourier_spectrum.png", fig)


def simulate_online_harmonic(
    *,
    method: str,
    model: ToyModelConfig,
    config: OnlineExperimentConfig,
    rng: np.random.Generator,
    kappa_override: float | None = None,
    k0_override: float | None = None,
) -> dict[str, np.ndarray]:
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"Unsupported method {method!r}")
    kappa = model.kappa if kappa_override is None else kappa_override
    k0 = model.k0 if k0_override is None else k0_override
    alpha = config.alpha
    gamma = gamma_from_alpha(alpha)
    grid = periodic_grid(config.grid_size)
    dz = periodic_dz(config.grid_size)
    modes = spectral_modes(config.grid_size)
    target_density = harmonic_tempered_density(grid, beta=model.beta, delta=model.delta, alpha=alpha)
    target_bias = harmonic_true_bias(grid, alpha=alpha, delta=model.delta)

    z = wrap_periodic(
        rng.normal(loc=config.init_z_center, scale=config.init_z_std, size=(config.num_replicates, config.num_walkers))
    )
    y = rng.normal(scale=config.init_y_std, size=(config.num_replicates, config.num_walkers))
    noise_scale = math.sqrt(2.0 * config.dt / model.beta)
    num_saves = config.num_steps // config.save_stride + 1
    times = np.zeros(num_saves, dtype=np.float64)
    metrics = {
        "force_error": np.zeros((num_saves, config.num_replicates), dtype=np.float64),
        "kl": np.zeros((num_saves, config.num_replicates), dtype=np.float64),
        "roughness": np.zeros((num_saves, config.num_replicates), dtype=np.float64),
        "transitions": np.zeros((num_saves, config.num_replicates), dtype=np.float64),
        "channel_fraction": np.zeros((num_saves, config.num_replicates), dtype=np.float64),
    }
    density_snapshots = np.zeros((num_saves, config.grid_size), dtype=np.float64)
    density_snapshot_std = np.zeros_like(density_snapshots)
    bias_snapshots = np.zeros_like(density_snapshots)
    bias_snapshot_std = np.zeros_like(density_snapshots)

    num_bins, _ = periodic_bin_count(config.bandwidth)
    bin_lookup = periodic_bin_indices(grid[None, :], num_bins)[0]
    cumulative_counts = np.zeros((config.num_replicates, num_bins), dtype=np.float64)
    cumulative_sums = np.zeros_like(cumulative_counts)
    transition_counts = np.zeros((config.num_replicates, config.num_walkers), dtype=np.float64)
    previous_wells = (np.cos(z) < 0.0).astype(np.int64)

    save_idx = 0
    for step in range(config.num_steps + 1):
        local_force = harmonic_local_force(
            z,
            y,
            beta=model.beta,
            delta=model.delta,
            kappa=kappa,
            k0=k0,
        )
        if method == "abf":
            sample_bins = periodic_bin_indices(z, num_bins)
            current_counts = batched_bincount(sample_bins, num_bins)
            current_sums = batched_bincount(sample_bins, num_bins, values=local_force)
            cumulative_counts += current_counts
            cumulative_sums += current_sums
            running_means = np.divide(
                cumulative_sums,
                cumulative_counts,
                out=np.zeros_like(cumulative_sums),
                where=cumulative_counts > 0.0,
            )
            bias_grid = gamma * running_means[:, bin_lookup]
            bias_walkers = gamma * running_means[
                np.broadcast_to(np.arange(config.num_replicates)[:, None], sample_bins.shape),
                sample_bins,
            ]
            density_estimate, _ = density_from_samples(
                z,
                bandwidth=config.bandwidth,
                grid_size=config.grid_size,
                dz=dz,
                modes=modes,
            )
        else:
            density_estimate, density_derivative = density_from_samples(
                z,
                bandwidth=config.bandwidth,
                grid_size=config.grid_size,
                dz=dz,
                modes=modes,
            )
            bias_grid = -(alpha / model.beta) * density_derivative / np.maximum(density_estimate, 1e-12)
            bias_walkers = interpolate_periodic_grid(bias_grid, z)

        if step % config.save_stride == 0:
            current_time = step * config.dt
            force_error = weighted_ise(bias_grid, target_bias, target_density, dz)
            density_safe = np.maximum(density_estimate, 1e-12)
            kl_value = periodic_integral(density_safe * (np.log(density_safe) - np.log(target_density[None, :])), dz)
            roughness = periodic_integral(spectral_derivative(bias_grid, modes) ** 2, dz)
            times[save_idx] = current_time
            metrics["force_error"][save_idx] = force_error
            metrics["kl"][save_idx] = kl_value
            metrics["roughness"][save_idx] = roughness
            metrics["transitions"][save_idx] = transition_counts.mean(axis=1)
            metrics["channel_fraction"][save_idx] = np.mean(y > 0.0, axis=1)
            density_snapshots[save_idx] = density_estimate.mean(axis=0)
            density_snapshot_std[save_idx] = density_estimate.std(axis=0)
            bias_snapshots[save_idx] = bias_grid.mean(axis=0)
            bias_snapshot_std[save_idx] = bias_grid.std(axis=0)
            save_idx += 1

        if step == config.num_steps:
            break

        y_drift = harmonic_y_drift(z, y, kappa=kappa, k0=k0)
        z = wrap_periodic(z + config.dt * (-local_force + bias_walkers) + noise_scale * rng.normal(size=z.shape))
        y = y + config.dt * y_drift + noise_scale * rng.normal(size=y.shape)
        current_wells = (np.cos(z) < 0.0).astype(np.int64)
        transition_counts += current_wells != previous_wells
        previous_wells = current_wells

    return {
        "times": times,
        "grid": grid,
        "target_density": target_density,
        "target_bias": target_bias,
        "density_mean": density_snapshots,
        "density_std": density_snapshot_std,
        "bias_mean": bias_snapshots,
        "bias_std": bias_snapshot_std,
        **metrics,
    }


def save_online_payload(
    output_dir: Path,
    method: str,
    data: dict[str, np.ndarray],
    *,
    config_json: str,
    model_json: str,
) -> None:
    payload = {key: value for key, value in data.items()}
    payload["method"] = np.asarray(method)
    payload["config_json"] = np.asarray(config_json)
    payload["model_json"] = np.asarray(model_json)
    np.savez_compressed(output_dir / f"{method}_online_data.npz", **payload)


def make_online_plots(
    output_dir: Path,
    harmonic_results: dict[str, dict[str, np.ndarray]],
) -> None:
    times = harmonic_results["abf"]["times"]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6), constrained_layout=True)
    metric_map = {
        "force_error": (axes[0, 0], r"$E_G(t)$"),
        "kl": (axes[0, 1], r"$H(p_t \mid \pi_\alpha)$"),
        "transitions": (axes[1, 0], "Mean well transitions"),
        "roughness": (axes[1, 1], r"$R_G(t)$"),
    }
    for metric, (ax, ylabel) in metric_map.items():
        for method in METHODS:
            values = harmonic_results[method][metric]
            plot_with_band(
                ax,
                times,
                values.mean(axis=1),
                values.std(axis=1),
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(loc="upper right")
        if metric in {"force_error", "kl", "roughness"}:
            ax.set_yscale("log")
    save_figure(output_dir / "online_convergence.png", fig)

    snapshot_indices = np.unique(np.linspace(0, times.size - 1, min(4, times.size), dtype=int))
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4), constrained_layout=True, sharex=True)
    target_density = harmonic_results["abf"]["target_density"]
    target_bias = harmonic_results["abf"]["target_bias"]
    grid = harmonic_results["abf"]["grid"]
    for row_idx, (field, target, title) in enumerate(
        (
            ("density_mean", target_density, "Marginal density snapshots"),
            ("bias_mean", target_bias, "Bias-force snapshots"),
        )
    ):
        for col_idx, method in enumerate(METHODS):
            ax = axes[row_idx, col_idx]
            ax.plot(grid, target, color="black", linewidth=2.2, label="Target")
            for snapshot_idx in snapshot_indices:
                ax.plot(
                    grid,
                    harmonic_results[method][field][snapshot_idx],
                    linewidth=1.6,
                    alpha=0.75,
                    label=f"t={times[snapshot_idx]:.2f}",
                )
            ax.set_title(f"{METHOD_LABELS[method]} {title}")
            ax.set_xlabel("z")
            ax.set_ylabel(title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=min(5, len(labels)))
    save_figure(output_dir / "online_snapshots.png", fig)


def run_online_experiment(
    model: ToyModelConfig,
    config: OnlineExperimentConfig,
    output_dir: Path,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_json = json.dumps(asdict(config), sort_keys=True)
    model_json = json.dumps(asdict(model), sort_keys=True)
    results: dict[str, dict[str, np.ndarray]] = {}
    for method_idx, method in enumerate(METHODS):
        method_rng = np.random.default_rng(seed + 101 * (method_idx + 1))
        results[method] = simulate_online_harmonic(method=method, model=model, config=config, rng=method_rng)
        save_online_payload(output_dir, method, results[method], config_json=config_json, model_json=model_json)
    make_online_plots(output_dir, results)
    return results


def simulate_harmonic_history_under_true_bias(
    *,
    model: ToyModelConfig,
    config: MemoryWindowExperimentConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    gamma = gamma_from_alpha(config.alpha)
    z = wrap_periodic(
        rng.normal(loc=config.init_z_center, scale=config.init_z_std, size=(config.num_replicates, config.num_walkers))
    )
    y = rng.normal(scale=config.init_y_std, size=(config.num_replicates, config.num_walkers))
    noise_scale = math.sqrt(2.0 * config.dt / model.beta)
    history_indices = np.arange(0, config.num_steps + 1, config.history_stride, dtype=int)
    z_history = np.zeros((history_indices.size, config.num_replicates, config.num_walkers), dtype=np.float64)
    force_history = np.zeros_like(z_history)
    times = history_indices * config.dt
    history_pos = 0

    for step in range(config.num_steps + 1):
        local_force = harmonic_local_force(
            z,
            y,
            beta=model.beta,
            delta=model.delta,
            kappa=model.kappa,
            k0=model.k0,
        )
        if step % config.history_stride == 0:
            z_history[history_pos] = z
            force_history[history_pos] = local_force
            history_pos += 1
        if step == config.num_steps:
            break
        true_bias = gamma * a0_prime(z, model.delta)
        y_drift = harmonic_y_drift(z, y, kappa=model.kappa, k0=model.k0)
        z = wrap_periodic(z + config.dt * (-local_force + true_bias) + noise_scale * rng.normal(size=z.shape))
        y = y + config.dt * y_drift + noise_scale * rng.normal(size=y.shape)

    return {
        "times": times,
        "z_history": z_history,
        "force_history": force_history,
    }


def run_memory_window_experiment(
    model: ToyModelConfig,
    config: MemoryWindowExperimentConfig,
    output_dir: Path,
    seed: int,
) -> dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    history = simulate_harmonic_history_under_true_bias(model=model, config=config, rng=rng)
    times = history["times"]
    z_history = history["z_history"]
    force_history = history["force_history"]
    grid = periodic_grid(config.grid_size)
    dz = periodic_dz(config.grid_size)
    modes = spectral_modes(config.grid_size)
    target_density = harmonic_tempered_density(grid, beta=model.beta, delta=model.delta, alpha=config.alpha)
    target_bias = harmonic_true_bias(grid, alpha=config.alpha, delta=model.delta)
    evaluation_idx = int(np.argmin(np.abs(times - config.evaluation_time)))
    weighted_ise_values = np.zeros((len(METHODS), len(config.window_sizes), config.num_replicates), dtype=np.float64)

    for window_idx, window_size in enumerate(config.window_sizes):
        frame_count = max(1, int(round(window_size / (config.dt * config.history_stride))))
        start_idx = max(0, evaluation_idx - frame_count + 1)
        z_window = z_history[start_idx : evaluation_idx + 1]
        force_window = force_history[start_idx : evaluation_idx + 1]
        flattened_z = np.transpose(z_window, (1, 0, 2)).reshape(config.num_replicates, -1)
        flattened_force = np.transpose(force_window, (1, 0, 2)).reshape(config.num_replicates, -1)
        abf_force, _, _ = abf_force_from_samples(
            flattened_z,
            flattened_force,
            gamma=gamma_from_alpha(config.alpha),
            bandwidth=config.bandwidth,
            grid=grid,
        )
        abp_force, _ = abp_force_from_samples(
            flattened_z,
            alpha=config.alpha,
            beta=model.beta,
            bandwidth=config.bandwidth,
            grid_size=config.grid_size,
            dz=dz,
            modes=modes,
        )
        weighted_ise_values[METHODS.index("abf"), window_idx] = weighted_ise(abf_force, target_bias, target_density, dz)
        weighted_ise_values[METHODS.index("abp"), window_idx] = weighted_ise(abp_force, target_bias, target_density, dz)

    density_snapshots = []
    density_times = []
    for snapshot_idx in np.unique(np.linspace(0, times.size - 1, min(4, times.size), dtype=int)):
        density_estimate, _ = density_from_samples(
            z_history[snapshot_idx],
            bandwidth=config.bandwidth,
            grid_size=config.grid_size,
            dz=dz,
            modes=modes,
        )
        density_snapshots.append(density_estimate.mean(axis=0))
        density_times.append(times[snapshot_idx])

    payload = {
        "grid": grid,
        "times": times,
        "evaluation_time": np.asarray(times[evaluation_idx]),
        "window_sizes": np.asarray(config.window_sizes),
        "weighted_ise": weighted_ise_values,
        "target_density": target_density,
        "target_bias": target_bias,
        "density_snapshots": np.stack(density_snapshots, axis=0),
        "density_snapshot_times": np.asarray(density_times),
        "config_json": np.asarray(json.dumps(asdict(config), sort_keys=True)),
        "model_json": np.asarray(json.dumps(asdict(model), sort_keys=True)),
    }
    np.savez_compressed(output_dir / "memory_window_data.npz", **payload)
    make_memory_window_plots(payload, output_dir)
    return payload


def make_memory_window_plots(data: dict[str, np.ndarray], output_dir: Path) -> None:
    window_sizes = np.asarray(data["window_sizes"])
    weighted_ise_values = np.asarray(data["weighted_ise"])
    grid = np.asarray(data["grid"])
    target_density = np.asarray(data["target_density"])
    density_snapshots = np.asarray(data["density_snapshots"])
    density_times = np.asarray(data["density_snapshot_times"])

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for method in METHODS:
        values = weighted_ise_values[METHODS.index(method)]
        plot_with_band(
            ax,
            window_sizes,
            values.mean(axis=1),
            values.std(axis=1),
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    ax.set_title("Window-Length Sweep During a Common Transient")
    ax.set_xlabel(r"Window length $T_w$")
    ax.set_ylabel("Weighted ISE")
    ax.legend(loc="upper right")
    save_figure(output_dir / "memory_window_error.png", fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    ax.plot(grid, target_density, color="black", linewidth=2.3, label="Target")
    for snapshot, snapshot_time in zip(density_snapshots, density_times):
        ax.plot(grid, snapshot, linewidth=1.8, alpha=0.8, label=f"t={snapshot_time:.2f}")
    ax.set_title("Transient Marginal Densities Used in the Window Study")
    ax.set_xlabel("z")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    save_figure(output_dir / "memory_window_density.png", fig)


def run_k0_control_experiment(
    base_model: ToyModelConfig,
    config: K0ControlExperimentConfig,
    output_dir: Path,
    seed: int,
) -> dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_force_error = np.zeros((len(METHODS), len(config.kappas), len(config.k0_values), config.num_replicates))
    final_kl = np.zeros_like(final_force_error)
    final_roughness = np.zeros_like(final_force_error)

    for kappa_idx, kappa in enumerate(config.kappas):
        for k0_idx, k0 in enumerate(config.k0_values):
            online_config = OnlineExperimentConfig(
                num_replicates=config.num_replicates,
                num_walkers=config.num_walkers,
                num_steps=config.num_steps,
                dt=config.dt,
                alpha=config.alpha,
                bandwidth=config.bandwidth,
                grid_size=config.grid_size,
                save_stride=config.save_stride,
                init_z_center=config.init_z_center,
                init_z_std=config.init_z_std,
                init_y_std=config.init_y_std,
            )
            for method_idx, method in enumerate(METHODS):
                method_rng = np.random.default_rng(seed + 1000 * (kappa_idx + 1) + 100 * (k0_idx + 1) + 11 * method_idx)
                result = simulate_online_harmonic(
                    method=method,
                    model=base_model,
                    config=online_config,
                    rng=method_rng,
                    kappa_override=kappa,
                    k0_override=k0,
                )
                final_force_error[method_idx, kappa_idx, k0_idx] = result["force_error"][-1]
                final_kl[method_idx, kappa_idx, k0_idx] = result["kl"][-1]
                final_roughness[method_idx, kappa_idx, k0_idx] = result["roughness"][-1]

    payload = {
        "kappas": np.asarray(config.kappas),
        "k0_values": np.asarray(config.k0_values),
        "final_force_error": final_force_error,
        "final_kl": final_kl,
        "final_roughness": final_roughness,
        "config_json": np.asarray(json.dumps(asdict(config), sort_keys=True)),
        "model_json": np.asarray(json.dumps(asdict(base_model), sort_keys=True)),
    }
    np.savez_compressed(output_dir / "k0_control_data.npz", **payload)
    make_k0_control_plot(payload, output_dir)
    return payload


def make_k0_control_plot(data: dict[str, np.ndarray], output_dir: Path) -> None:
    kappas = np.asarray(data["kappas"])
    k0_values = np.asarray(data["k0_values"])
    final_force_error = np.asarray(data["final_force_error"])
    final_kl = np.asarray(data["final_kl"])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for metric, ax, ylabel in (
        (final_force_error, axes[0], "Final force error"),
        (final_kl, axes[1], "Final KL divergence"),
    ):
        for kappa_idx, kappa in enumerate(kappas):
            for method in METHODS:
                values = metric[METHODS.index(method), kappa_idx]
                label = f"{METHOD_LABELS[method]}, kappa={kappa:g}"
                plot_with_band(
                    ax,
                    k0_values,
                    values.mean(axis=1),
                    values.std(axis=1),
                    label=label,
                    color=METHOD_COLORS[method],
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Orthogonal stiffness $k_0$")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
    save_figure(output_dir / "orthogonal_relaxation_control.png", fig)


def hidden_barrier_local_force(
    z: np.ndarray,
    y: np.ndarray,
    *,
    delta: float,
    coupling: float,
) -> np.ndarray:
    return a0_prime(z, delta) + coupling * y * np.cos(z)


def hidden_barrier_y_drift(
    z: np.ndarray,
    y: np.ndarray,
    *,
    fiber_lambda: float,
    coupling: float,
) -> np.ndarray:
    return -(4.0 * fiber_lambda * y * (y**2 - 1.0) + coupling * np.sin(z))


def hidden_barrier_reference(
    grid: np.ndarray,
    *,
    alpha: float,
    beta: float,
    delta: float,
    fiber_lambda: float,
    coupling: float,
    y_max: float,
    y_grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_grid = np.linspace(-y_max, y_max, y_grid_size)
    z_mesh = grid[:, None]
    y_mesh = y_grid[None, :]
    potential = a0(z_mesh, delta) + fiber_lambda * (y_mesh**2 - 1.0) ** 2 + coupling * y_mesh * np.sin(z_mesh)
    local_force = hidden_barrier_local_force(z_mesh, y_mesh, delta=delta, coupling=coupling)
    log_weights = -beta * potential
    shifts = log_weights.max(axis=1, keepdims=True)
    weights = np.exp(log_weights - shifts)
    partition = np.trapezoid(weights, y_grid, axis=1)
    force_numerator = np.trapezoid(local_force * weights, y_grid, axis=1)
    free_energy = -(np.log(np.maximum(partition, 1e-300)) + shifts[:, 0]) / beta
    free_energy -= free_energy.min()
    mean_force = force_numerator / np.maximum(partition, 1e-300)
    marginal = np.exp(-beta * free_energy / (1.0 + alpha))
    dz = periodic_dz(grid.size)
    marginal /= periodic_integral(marginal, dz)
    return free_energy, mean_force, marginal


def simulate_online_hidden_barrier(
    *,
    method: str,
    config: HiddenBarrierExperimentConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    method = method.lower()
    gamma = gamma_from_alpha(config.alpha)
    grid = periodic_grid(config.grid_size)
    dz = periodic_dz(config.grid_size)
    modes = spectral_modes(config.grid_size)
    _, target_force, target_density = hidden_barrier_reference(
        grid,
        alpha=config.alpha,
        beta=config.beta,
        delta=config.delta,
        fiber_lambda=config.fiber_lambda,
        coupling=config.coupling,
        y_max=config.y_max,
        y_grid_size=config.y_grid_size,
    )
    target_bias = gamma * target_force
    z = wrap_periodic(
        rng.normal(loc=config.init_z_center, scale=config.init_z_std, size=(config.num_replicates, config.num_walkers))
    )
    y = rng.normal(loc=config.init_y_center, scale=config.init_y_std, size=(config.num_replicates, config.num_walkers))
    noise_scale = math.sqrt(2.0 * config.dt / config.beta)
    num_saves = config.num_steps // config.save_stride + 1
    times = np.zeros(num_saves, dtype=np.float64)
    force_error = np.zeros((num_saves, config.num_replicates), dtype=np.float64)
    channel_fraction = np.zeros_like(force_error)
    density_mean = np.zeros((num_saves, config.grid_size), dtype=np.float64)
    bias_mean = np.zeros_like(density_mean)

    num_bins, _ = periodic_bin_count(config.bandwidth)
    bin_lookup = periodic_bin_indices(grid[None, :], num_bins)[0]
    cumulative_counts = np.zeros((config.num_replicates, num_bins), dtype=np.float64)
    cumulative_sums = np.zeros_like(cumulative_counts)
    save_idx = 0

    for step in range(config.num_steps + 1):
        local_force = hidden_barrier_local_force(z, y, delta=config.delta, coupling=config.coupling)
        if method == "abf":
            sample_bins = periodic_bin_indices(z, num_bins)
            cumulative_counts += batched_bincount(sample_bins, num_bins)
            cumulative_sums += batched_bincount(sample_bins, num_bins, values=local_force)
            running_means = np.divide(
                cumulative_sums,
                cumulative_counts,
                out=np.zeros_like(cumulative_sums),
                where=cumulative_counts > 0.0,
            )
            bias_grid = gamma * running_means[:, bin_lookup]
            bias_walkers = gamma * running_means[
                np.broadcast_to(np.arange(config.num_replicates)[:, None], sample_bins.shape),
                sample_bins,
            ]
        else:
            density, density_derivative = density_from_samples(
                z,
                bandwidth=config.bandwidth,
                grid_size=config.grid_size,
                dz=dz,
                modes=modes,
            )
            bias_grid = -(config.alpha / config.beta) * density_derivative / np.maximum(density, 1e-12)
            bias_walkers = interpolate_periodic_grid(bias_grid, z)

        if step % config.save_stride == 0:
            density, _ = density_from_samples(
                z,
                bandwidth=config.bandwidth,
                grid_size=config.grid_size,
                dz=dz,
                modes=modes,
            )
            force_error[save_idx] = weighted_ise(bias_grid, target_bias, target_density, dz)
            channel_fraction[save_idx] = np.mean(y > 0.0, axis=1)
            density_mean[save_idx] = density.mean(axis=0)
            bias_mean[save_idx] = bias_grid.mean(axis=0)
            times[save_idx] = step * config.dt
            save_idx += 1

        if step == config.num_steps:
            break

        y_drift = hidden_barrier_y_drift(
            z,
            y,
            fiber_lambda=config.fiber_lambda,
            coupling=config.coupling,
        )
        z = wrap_periodic(z + config.dt * (-local_force + bias_walkers) + noise_scale * rng.normal(size=z.shape))
        y = y + config.dt * y_drift + noise_scale * rng.normal(size=y.shape)

    return {
        "times": times,
        "grid": grid,
        "target_density": target_density,
        "target_bias": target_bias,
        "force_error": force_error,
        "channel_fraction": channel_fraction,
        "density_mean": density_mean,
        "bias_mean": bias_mean,
    }


def run_hidden_barrier_experiment(
    config: HiddenBarrierExperimentConfig,
    output_dir: Path,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, np.ndarray]] = {}
    for method_idx, method in enumerate(METHODS):
        method_rng = np.random.default_rng(seed + 211 * (method_idx + 1))
        results[method] = simulate_online_hidden_barrier(method=method, config=config, rng=method_rng)
        payload = {key: value for key, value in results[method].items()}
        payload["method"] = np.asarray(method)
        payload["config_json"] = np.asarray(json.dumps(asdict(config), sort_keys=True))
        np.savez_compressed(output_dir / f"{method}_hidden_barrier_data.npz", **payload)
    make_hidden_barrier_plots(output_dir, results)
    return results


def make_hidden_barrier_plots(
    output_dir: Path,
    results: dict[str, dict[str, np.ndarray]],
) -> None:
    grid = results["abf"]["grid"]
    target_bias = results["abf"]["target_bias"]
    times = results["abf"]["times"]
    final_idx = -1

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    axes[0].plot(grid, target_bias, color="black", linewidth=2.2, label="Target")
    for method in METHODS:
        axes[0].plot(
            grid,
            results[method]["bias_mean"][final_idx],
            linewidth=2.0,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[0].set_title("Final Bias Under Hidden Orthogonal Barriers")
    axes[0].set_xlabel("z")
    axes[0].set_ylabel(r"$\widehat G(z)$")
    axes[0].legend(loc="upper right")

    for method in METHODS:
        channel_fraction = results[method]["channel_fraction"]
        plot_with_band(
            axes[1],
            times,
            channel_fraction.mean(axis=1),
            channel_fraction.std(axis=1),
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    axes[1].set_title("Occupancy of the Positive y-Channel")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(r"$\mathbb{P}(Y_t > 0)$")
    axes[1].legend(loc="best")
    save_figure(output_dir / "hidden_barrier_summary.png", fig)


def make_suite_overview(
    static_data: dict[str, np.ndarray] | None,
    online_data: dict[str, dict[str, np.ndarray]] | None,
    static_config: StaticExperimentConfig | None,
    output_dir: Path,
) -> None:
    if static_data is None or online_data is None or static_config is None:
        return
    alpha_idx, kappa_idx, sample_idx, _ = representative_indices(static_config)
    sample_sizes = np.asarray(static_data["sample_sizes"])
    bandwidths = np.asarray(static_data["bandwidths"])
    actual_abf_bandwidths = np.asarray(static_data["actual_abf_bandwidths"])
    weighted_variance = np.asarray(static_data["weighted_variance"])
    weighted_ise_values = np.asarray(static_data["weighted_ise"])
    overlay_truth = np.asarray(static_data["overlay_truth"])
    overlay_abf = np.asarray(static_data["overlay_abf"])
    overlay_abp = np.asarray(static_data["overlay_abp"])
    grid = np.asarray(static_data["grid"])

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.9), constrained_layout=True)
    axes[0, 0].plot(grid, overlay_truth, color="black", linewidth=2.3, label=r"$G^\star$")
    for curve in overlay_abf:
        axes[0, 0].plot(grid, curve, color=METHOD_COLORS["abf"], alpha=0.3, linewidth=1.0)
    for curve in overlay_abp:
        axes[0, 0].plot(grid, curve, color=METHOD_COLORS["abp"], alpha=0.3, linewidth=1.0)
    axes[0, 0].set_title("Force Estimates at Fixed N")
    axes[0, 0].set_xlabel("z")
    axes[0, 0].set_ylabel(r"$\widehat G(z)$")

    representative_weighted_variance = weighted_variance[:, alpha_idx, kappa_idx, sample_idx]
    axes[0, 1].loglog(
        actual_abf_bandwidths,
        representative_weighted_variance[METHODS.index("abf")],
        marker="o",
        linewidth=2.0,
        color=METHOD_COLORS["abf"],
        label="ABF",
    )
    axes[0, 1].loglog(
        bandwidths,
        representative_weighted_variance[METHODS.index("abp")],
        marker="s",
        linewidth=2.0,
        color=METHOD_COLORS["abp"],
        label="ABP",
    )
    axes[0, 1].set_title("Variance Against Bandwidth")
    axes[0, 1].set_xlabel("Bandwidth h")
    axes[0, 1].set_ylabel(r"$\int \mathrm{Var}(\widehat G)\,\pi_\alpha\,dz$")
    axes[0, 1].legend(loc="lower left")

    best_weighted_ise = weighted_ise_values.mean(axis=-1).min(axis=-1)
    for method in METHODS:
        method_idx = METHODS.index(method)
        axes[1, 0].loglog(
            sample_sizes,
            best_weighted_ise[method_idx, alpha_idx, kappa_idx],
            marker="o",
            linewidth=2.0,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[1, 0].set_title("Best Force MSE Against N")
    axes[1, 0].set_xlabel("Sample size N")
    axes[1, 0].set_ylabel("Best weighted ISE")
    axes[1, 0].legend(loc="upper right")

    ax = axes[1, 1]
    ax2 = ax.twinx()
    times = online_data["abf"]["times"]
    for method in METHODS:
        force_mean = online_data[method]["force_error"].mean(axis=1)
        kl_mean = online_data[method]["kl"].mean(axis=1)
        ax.plot(times, force_mean, color=METHOD_COLORS[method], linewidth=2.0, label=f"{METHOD_LABELS[method]} force error")
        ax2.plot(times, kl_mean, color=METHOD_COLORS[method], linewidth=1.8, linestyle="--", label=f"{METHOD_LABELS[method]} KL")
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_title("Online Convergence")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$E_G(t)$")
    ax2.set_ylabel(r"$H(p_t \mid \pi_\alpha)$")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    save_figure(output_dir / "suite_overview.png", fig)


def build_profile(name: str) -> SuiteProfile:
    if name == "smoke":
        return SuiteProfile(
            model=ToyModelConfig(beta=1.0, delta=6.0, k0=20.0, kappa=1.0),
            static=StaticExperimentConfig(
                num_repeats=64,
                batch_size=16,
                sample_sizes=(128, 512, 2048),
                bandwidths=(0.03, 0.08, 0.18),
                alphas=(1.0, 4.0),
                kappas=(0.0, 1.0, 2.0),
                grid_size=512,
                force_overlay_repeats=6,
                representative_alpha=4.0,
                representative_kappa=1.0,
                representative_sample_size=512,
                representative_bandwidth=0.08,
            ),
            fourier=FourierExperimentConfig(
                num_repeats=128,
                batch_size=32,
                sample_size=2048,
                alpha=4.0,
                kappa=1.0,
                bandwidths=(0.03, 0.08, 0.18),
                grid_size=512,
            ),
            online=OnlineExperimentConfig(
                num_replicates=8,
                num_walkers=48,
                num_steps=2_000,
                dt=5e-4,
                alpha=4.0,
                bandwidth=0.08,
                grid_size=256,
                save_stride=20,
                init_z_center=0.0,
                init_z_std=0.08,
                init_y_std=0.12,
            ),
            memory=MemoryWindowExperimentConfig(
                num_replicates=8,
                num_walkers=32,
                num_steps=2_000,
                dt=5e-4,
                alpha=4.0,
                bandwidth=0.08,
                grid_size=256,
                history_stride=5,
                evaluation_time=0.6,
                window_sizes=(0.05, 0.1, 0.2, 0.4, 0.8),
                init_z_center=0.0,
                init_z_std=0.08,
                init_y_std=0.12,
            ),
            k0_control=K0ControlExperimentConfig(
                num_replicates=6,
                num_walkers=48,
                num_steps=1_800,
                dt=5e-4,
                alpha=4.0,
                bandwidth=0.08,
                grid_size=256,
                save_stride=30,
                kappas=(0.5, 1.5),
                k0_values=(5.0, 20.0, 50.0),
                init_z_center=0.0,
                init_z_std=0.08,
                init_y_std=0.12,
            ),
            hidden_barrier=HiddenBarrierExperimentConfig(
                num_replicates=6,
                num_walkers=48,
                num_steps=3_000,
                dt=4e-4,
                alpha=4.0,
                bandwidth=0.08,
                grid_size=256,
                save_stride=30,
                beta=1.0,
                delta=6.0,
                fiber_lambda=8.0,
                coupling=2.0,
                y_max=3.0,
                y_grid_size=2_001,
                init_z_center=0.0,
                init_z_std=0.08,
                init_y_center=1.0,
                init_y_std=0.08,
            ),
        )

    if name == "standard":
        return SuiteProfile(
            model=ToyModelConfig(beta=1.0, delta=6.0, k0=20.0, kappa=1.0),
            static=StaticExperimentConfig(
                num_repeats=160,
                batch_size=32,
                sample_sizes=(128, 256, 512, 1024, 2048, 4096),
                bandwidths=(0.02, 0.03, 0.05, 0.08, 0.12, 0.18),
                alphas=(1.0, 4.0, 9.0),
                kappas=(0.0, 0.5, 1.0, 2.0),
                grid_size=768,
                force_overlay_repeats=6,
                representative_alpha=4.0,
                representative_kappa=1.0,
                representative_sample_size=1024,
                representative_bandwidth=0.08,
            ),
            fourier=FourierExperimentConfig(
                num_repeats=256,
                batch_size=32,
                sample_size=4096,
                alpha=4.0,
                kappa=1.0,
                bandwidths=(0.02, 0.05, 0.08, 0.18),
                grid_size=768,
            ),
            online=OnlineExperimentConfig(
                num_replicates=16,
                num_walkers=64,
                num_steps=4_000,
                dt=5e-4,
                alpha=4.0,
                bandwidth=0.08,
                grid_size=384,
                save_stride=20,
                init_z_center=0.0,
                init_z_std=0.08,
                init_y_std=0.12,
            ),
            memory=MemoryWindowExperimentConfig(
                num_replicates=16,
                num_walkers=48,
                num_steps=4_000,
                dt=5e-4,
                alpha=4.0,
                bandwidth=0.08,
                grid_size=384,
                history_stride=5,
                evaluation_time=0.8,
                window_sizes=(0.05, 0.1, 0.2, 0.4, 0.8, 1.2),
                init_z_center=0.0,
                init_z_std=0.08,
                init_y_std=0.12,
            ),
            k0_control=K0ControlExperimentConfig(
                num_replicates=12,
                num_walkers=64,
                num_steps=3_000,
                dt=5e-4,
                alpha=4.0,
                bandwidth=0.08,
                grid_size=384,
                save_stride=25,
                kappas=(0.5, 1.5),
                k0_values=(5.0, 10.0, 20.0, 50.0),
                init_z_center=0.0,
                init_z_std=0.08,
                init_y_std=0.12,
            ),
            hidden_barrier=HiddenBarrierExperimentConfig(
                num_replicates=10,
                num_walkers=64,
                num_steps=5_000,
                dt=4e-4,
                alpha=4.0,
                bandwidth=0.08,
                grid_size=384,
                save_stride=25,
                beta=1.0,
                delta=6.0,
                fiber_lambda=8.0,
                coupling=2.0,
                y_max=3.0,
                y_grid_size=2_401,
                init_z_center=0.0,
                init_z_std=0.08,
                init_y_center=1.0,
                init_y_std=0.08,
            ),
        )

    if name == "paper":
        return SuiteProfile(
            model=ToyModelConfig(beta=1.0, delta=6.0, k0=20.0, kappa=1.0),
            static=StaticExperimentConfig(
                num_repeats=320,
                batch_size=32,
                sample_sizes=(128, 256, 512, 1024, 2048, 4096, 8192),
                bandwidths=(0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25),
                alphas=(1.0, 4.0, 9.0),
                kappas=(0.0, 0.5, 1.0, 2.0),
                grid_size=1024,
                force_overlay_repeats=8,
                representative_alpha=4.0,
                representative_kappa=1.0,
                representative_sample_size=1024,
                representative_bandwidth=0.08,
            ),
            fourier=FourierExperimentConfig(
                num_repeats=384,
                batch_size=32,
                sample_size=8192,
                alpha=4.0,
                kappa=1.0,
                bandwidths=(0.02, 0.05, 0.08, 0.18),
                grid_size=1024,
            ),
            online=OnlineExperimentConfig(
                num_replicates=24,
                num_walkers=96,
                num_steps=6_000,
                dt=4e-4,
                alpha=4.0,
                bandwidth=0.08,
                grid_size=512,
                save_stride=20,
                init_z_center=0.0,
                init_z_std=0.08,
                init_y_std=0.12,
            ),
            memory=MemoryWindowExperimentConfig(
                num_replicates=24,
                num_walkers=64,
                num_steps=6_000,
                dt=4e-4,
                alpha=4.0,
                bandwidth=0.08,
                grid_size=512,
                history_stride=5,
                evaluation_time=1.0,
                window_sizes=(0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6),
                init_z_center=0.0,
                init_z_std=0.08,
                init_y_std=0.12,
            ),
            k0_control=K0ControlExperimentConfig(
                num_replicates=20,
                num_walkers=96,
                num_steps=4_500,
                dt=4e-4,
                alpha=4.0,
                bandwidth=0.08,
                grid_size=512,
                save_stride=25,
                kappas=(0.5, 1.5),
                k0_values=(5.0, 10.0, 20.0, 50.0),
                init_z_center=0.0,
                init_z_std=0.08,
                init_y_std=0.12,
            ),
            hidden_barrier=HiddenBarrierExperimentConfig(
                num_replicates=16,
                num_walkers=96,
                num_steps=7_000,
                dt=3e-4,
                alpha=4.0,
                bandwidth=0.08,
                grid_size=512,
                save_stride=25,
                beta=1.0,
                delta=6.0,
                fiber_lambda=8.0,
                coupling=2.0,
                y_max=3.0,
                y_grid_size=2_801,
                init_z_center=0.0,
                init_z_std=0.08,
                init_y_center=1.0,
                init_y_std=0.08,
            ),
        )

    raise ValueError(f"Unsupported profile {name!r}")


def normalize_experiment_selection(values: Iterable[str]) -> list[str]:
    selected = [value.lower() for value in values]
    if "all" in selected:
        return ["static", "fourier", "online", "memory", "k0-control", "hidden-barrier"]
    return selected


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Toy-model ABF versus ABP experiments inside classic_md.")
    parser.add_argument(
        "--profile",
        choices=("smoke", "standard", "paper"),
        default="smoke",
        help="Preset controlling sweep sizes and runtime.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=("all",),
        choices=("all", "static", "fourier", "online", "memory", "k0-control", "hidden-barrier"),
        help="Subset of experiments to run.",
    )
    parser.add_argument("--exp-name", type=str, default=None, help="Result folder name under classic_md/results/.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "results")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    profile = build_profile(args.profile)
    experiments = normalize_experiment_selection(args.experiments)
    exp_name = args.exp_name or f"toy_abf_abp_{args.profile}"
    root_dir = args.output_root / exp_name
    root_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        root_dir / "manifest.json",
        {
            "profile": args.profile,
            "seed": args.seed,
            "experiments": experiments,
            "model": asdict(profile.model),
            "static": asdict(profile.static),
            "fourier": asdict(profile.fourier),
            "online": asdict(profile.online),
            "memory": asdict(profile.memory),
            "k0_control": asdict(profile.k0_control),
            "hidden_barrier": asdict(profile.hidden_barrier),
        },
    )

    static_data: dict[str, np.ndarray] | None = None
    online_data: dict[str, dict[str, np.ndarray]] | None = None

    if "static" in experiments:
        static_data = run_static_experiment(profile.model, profile.static, root_dir / "static", args.seed + 11)
    if "fourier" in experiments:
        run_fourier_experiment(profile.model, profile.fourier, root_dir / "fourier", args.seed + 23)
    if "online" in experiments:
        online_data = run_online_experiment(profile.model, profile.online, root_dir / "online", args.seed + 37)
    if "memory" in experiments:
        run_memory_window_experiment(profile.model, profile.memory, root_dir / "memory_window", args.seed + 47)
    if "k0-control" in experiments:
        run_k0_control_experiment(profile.model, profile.k0_control, root_dir / "orthogonal_relaxation", args.seed + 59)
    if "hidden-barrier" in experiments:
        run_hidden_barrier_experiment(profile.hidden_barrier, root_dir / "hidden_barrier", args.seed + 71)

    make_suite_overview(static_data, online_data, profile.static if static_data is not None else None, root_dir)
    print(f"Saved toy-model ABF/ABP experiments to {root_dir}")


if __name__ == "__main__":
    main()
