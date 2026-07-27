from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
SQRT_2PI = math.sqrt(2.0 * math.pi)
REFERENCE_Y_GRID_SIZE = 4097
REFERENCE_YMAX_FLOOR = 6.0
ROTATION_TOL = 1e-12


@dataclass(frozen=True)
class EntropicPotentialConfig:
    beta: float = 1.0
    kappa0: float = 1.0
    sigma: float = 0.25
    d: float = 2.0
    xmax: float = 2.5
    theta: float = 0.0


@dataclass(frozen=True)
class EstimatorConfig:
    grid_size: int = 256
    bandwidth: float = 0.08
    floor: float = 1e-8
    burn_in: float = 0.0


@dataclass(frozen=True)
class InitialConditionConfig:
    center_x: float = -1.0
    center_y: float = 0.0
    std: float = 0.05


@dataclass(frozen=True)
class ClassicalMDConfig:
    method: Literal["abf", "abp"] = "abf"
    num_trajectories: int = 128
    num_steps: int = 10_000
    dt: float = 1e-3
    alpha: float = 5.0
    gamma: float | None = None
    seed: int = 0
    save_stride: int = 100
    store_particles: bool = False
    store_estimators: bool = False
    potential: EntropicPotentialConfig = field(default_factory=EntropicPotentialConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    initial: InitialConditionConfig = field(default_factory=InitialConditionConfig)


@dataclass
class SimulationResult:
    method: str
    alpha: float
    gamma: float
    x_grid: np.ndarray
    times: np.ndarray
    free_energy: np.ndarray
    free_energy_gradient: np.ndarray
    marginal_mean: np.ndarray
    marginal_var: np.ndarray
    mean_force_mean: np.ndarray
    mean_force_var: np.ndarray
    bias_mean: np.ndarray
    bias_var: np.ndarray
    instant_marginal_mean: np.ndarray
    instant_marginal_var: np.ndarray
    instant_mean_force_mean: np.ndarray
    instant_mean_force_var: np.ndarray
    particles: np.ndarray | None = None
    marginal_estimates: np.ndarray | None = None
    mean_force_estimates: np.ndarray | None = None
    bias_estimates: np.ndarray | None = None


class EstimatorState(NamedTuple):
    marginal: Array
    marginal_dx: Array
    mean_force_num: Array
    count: Array


class SimulationState(NamedTuple):
    positions: Array
    estimator: EstimatorState
    key: Array
    step: Array


class Observation(NamedTuple):
    positions: Array
    marginal: Array
    mean_force: Array
    bias: Array
    instant_marginal: Array
    instant_mean_force: Array


def stiffness_profile(x: Array, config: EntropicPotentialConfig) -> Array:
    return 1.0 + config.kappa0 * jnp.exp(-(x**2) / (2.0 * config.sigma**2))


def stiffness_profile_derivative(x: Array, config: EntropicPotentialConfig) -> Array:
    return -(config.kappa0 * x / config.sigma**2) * jnp.exp(-(x**2) / (2.0 * config.sigma**2))


def effective_transverse_stiffness(x: Array, config: EntropicPotentialConfig) -> Array:
    return jnp.exp((config.d - 1.0) * jnp.log(stiffness_profile(x, config)))


def effective_transverse_stiffness_derivative(x: Array, config: EntropicPotentialConfig) -> Array:
    k_profile = stiffness_profile(x, config)
    return effective_transverse_stiffness(x, config) * (config.d - 1.0) * stiffness_profile_derivative(x, config) / k_profile


def rotate_to_potential_frame(points: Array, theta: float) -> Array:
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    x = points[..., 0]
    y = points[..., 1]
    rotated_x = cos_theta * x + sin_theta * y
    rotated_y = -sin_theta * x + cos_theta * y
    return jnp.stack((rotated_x, rotated_y), axis=-1)


def rotate_from_potential_frame(vectors: Array, theta: float) -> Array:
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    x = vectors[..., 0]
    y = vectors[..., 1]
    lab_x = cos_theta * x - sin_theta * y
    lab_y = sin_theta * x + cos_theta * y
    return jnp.stack((lab_x, lab_y), axis=-1)


def base_entropic_potential(points: Array, config: EntropicPotentialConfig) -> Array:
    x = points[..., 0]
    y = points[..., 1]
    return (x**2 - 1.0) ** 2 + effective_transverse_stiffness(x, config) * y**2


def base_entropic_potential_gradient(points: Array, config: EntropicPotentialConfig) -> Array:
    x = points[..., 0]
    y = points[..., 1]
    alpha_eff = effective_transverse_stiffness(x, config)
    alpha_eff_prime = effective_transverse_stiffness_derivative(x, config)
    grad_x = 4.0 * x * (x**2 - 1.0) + alpha_eff_prime * y**2
    grad_y = 2.0 * alpha_eff * y
    return jnp.stack((grad_x, grad_y), axis=-1)


def entropic_potential(points: Array, config: EntropicPotentialConfig) -> Array:
    rotated_points = rotate_to_potential_frame(points, config.theta)
    return base_entropic_potential(rotated_points, config)


def entropic_potential_gradient(points: Array, config: EntropicPotentialConfig) -> Array:
    rotated_points = rotate_to_potential_frame(points, config.theta)
    rotated_gradient = base_entropic_potential_gradient(rotated_points, config)
    return rotate_from_potential_frame(rotated_gradient, config.theta)


def exact_free_energy(x: Array, config: EntropicPotentialConfig) -> Array:
    return (x**2 - 1.0) ** 2 + 0.5 * (config.d - 1.0) * jnp.log(stiffness_profile(x, config)) / config.beta


def exact_free_energy_gradient(x: Array, config: EntropicPotentialConfig) -> Array:
    k_profile = stiffness_profile(x, config)
    return 4.0 * x * (x**2 - 1.0) + 0.5 * (config.d - 1.0) * stiffness_profile_derivative(x, config) / (config.beta * k_profile)


def rotated_reference_y_grid(config: EntropicPotentialConfig) -> Array:
    ymax = max(REFERENCE_YMAX_FLOOR, 2.0 * config.xmax + 1.0)
    return jnp.linspace(-ymax, ymax, REFERENCE_Y_GRID_SIZE)


def numerical_free_energy_and_gradient(x: Array, config: EntropicPotentialConfig) -> tuple[Array, Array]:
    y_grid = rotated_reference_y_grid(config)
    x_mesh = jnp.broadcast_to(x[:, None], (x.shape[0], y_grid.shape[0]))
    y_mesh = jnp.broadcast_to(y_grid[None, :], (x.shape[0], y_grid.shape[0]))
    points = jnp.stack((x_mesh, y_mesh), axis=-1)
    energies = entropic_potential(points, config)
    log_weights = -config.beta * energies
    log_shift = jnp.max(log_weights, axis=1, keepdims=True)
    shifted_weights = jnp.exp(log_weights - log_shift)
    partition = jnp.trapezoid(shifted_weights, y_grid, axis=1)
    safe_partition = jnp.maximum(partition, jnp.finfo(shifted_weights.dtype).tiny)
    grad_x = entropic_potential_gradient(points, config)[..., 0]
    force_numerator = jnp.trapezoid(grad_x * shifted_weights, y_grid, axis=1)
    free_energy_values = -(jnp.log(safe_partition) + log_shift[:, 0]) / config.beta
    free_energy_values = free_energy_values - jnp.min(free_energy_values)
    return free_energy_values, force_numerator / safe_partition


def reference_free_energy_and_gradient(x: Array, config: EntropicPotentialConfig) -> tuple[Array, Array]:
    if abs(config.theta) <= ROTATION_TOL:
        return exact_free_energy(x, config), exact_free_energy_gradient(x, config)
    return numerical_free_energy_and_gradient(x, config)


def free_energy(x: Array, config: EntropicPotentialConfig) -> Array:
    values, _ = reference_free_energy_and_gradient(x, config)
    return values


def free_energy_gradient(x: Array, config: EntropicPotentialConfig) -> Array:
    _, gradient = reference_free_energy_and_gradient(x, config)
    return gradient


def stationary_gamma(method: str, alpha: float, gamma: float | None) -> float:
    if method.lower() == "abp":
        if not math.isfinite(alpha):
            raise ValueError("ABP requires a finite alpha.")
        return alpha / (1.0 + alpha)
    return resolve_gamma(alpha, gamma)


def stationary_marginal_density(x: Array, free_energy_values: Array, beta: float, gamma: float) -> Array:
    log_weights = -beta * (1.0 - gamma) * free_energy_values
    log_weights = log_weights - jnp.max(log_weights)
    weights = jnp.exp(log_weights)
    return weights / jnp.trapezoid(weights, x)


def stationary_bias(mean_force: Array, method: str, alpha: float, gamma: float) -> Array:
    gamma_eff = stationary_gamma(method, alpha, gamma if method.lower() == "abf" else None)
    return gamma_eff * mean_force


def resolve_gamma(alpha: float, gamma: float | None) -> float:
    if gamma is not None:
        return float(gamma)
    if math.isinf(alpha):
        return 1.0
    return alpha / (1.0 + alpha)


def make_x_grid(config: EntropicPotentialConfig, estimator: EstimatorConfig) -> Array:
    return jnp.linspace(-config.xmax, config.xmax, estimator.grid_size)


def gaussian_kernel(diff: Array, bandwidth: float) -> Array:
    scaled = diff / bandwidth
    return jnp.exp(-0.5 * scaled**2) / (SQRT_2PI * bandwidth)


def kernel_observables(
    positions: Array,
    grid: Array,
    potential: EntropicPotentialConfig,
    estimator: EstimatorConfig,
) -> tuple[Array, Array, Array]:
    x = positions[:, 0]
    diff = grid[None, :] - x[:, None]
    kernel = gaussian_kernel(diff, estimator.bandwidth)
    grad_x = entropic_potential_gradient(positions, potential)[:, 0]
    marginal = kernel
    marginal_dx = -(diff / (estimator.bandwidth**2)) * kernel
    mean_force_num = grad_x[:, None] * kernel
    return marginal, marginal_dx, mean_force_num


def interpolation_from_uniform_grid(grid: Array, values: Array, x: Array) -> Array:
    dx = grid[1] - grid[0]
    scaled = jnp.clip((x - grid[0]) / dx, 0.0, grid.size - 1.0)
    left = jnp.clip(jnp.floor(scaled).astype(jnp.int32), 0, grid.size - 2)
    weight = scaled - left.astype(scaled.dtype)
    return (1.0 - weight) * values[jnp.arange(values.shape[0]), left] + weight * values[jnp.arange(values.shape[0]), left + 1]


def bias_from_estimator(
    estimator_state: EstimatorState,
    method: str,
    alpha: float,
    gamma: float,
    beta: float,
    floor: float,
) -> tuple[Array, Array]:
    initialized = estimator_state.count > 0
    safe_marginal = jnp.maximum(estimator_state.marginal, floor)
    mean_force = estimator_state.mean_force_num / safe_marginal
    zero_bias = jnp.zeros_like(mean_force)
    if method == "abf":
        bias = gamma * mean_force
    else:
        bias = -(alpha / beta) * estimator_state.marginal_dx / safe_marginal
    bias = jnp.where(initialized, bias, zero_bias)
    mean_force = jnp.where(initialized, mean_force, zero_bias)
    return bias, mean_force


def make_initial_state(config: ClassicalMDConfig) -> SimulationState:
    key = jax.random.PRNGKey(config.seed)
    pos_key, next_key = jax.random.split(key)
    positions = jax.random.normal(pos_key, (config.num_trajectories, 2))
    positions = positions * config.initial.std + jnp.array([config.initial.center_x, config.initial.center_y])
    zeros = jnp.zeros((config.num_trajectories, config.estimator.grid_size))
    estimator_state = EstimatorState(
        marginal=zeros,
        marginal_dx=zeros,
        mean_force_num=zeros,
        count=jnp.array(0, dtype=jnp.int32),
    )
    return SimulationState(
        positions=positions,
        estimator=estimator_state,
        key=next_key,
        step=jnp.array(0, dtype=jnp.int32),
    )


def make_step_fn(config: ClassicalMDConfig):
    method = config.method.lower()
    potential = config.potential
    estimator = config.estimator
    grid = make_x_grid(potential, estimator)
    gamma = resolve_gamma(config.alpha, config.gamma)
    noise_scale = math.sqrt(2.0 * config.dt / potential.beta)

    def update_estimator(
        estimator_state: EstimatorState,
        sample_marginal: Array,
        sample_marginal_dx: Array,
        sample_mean_force_num: Array,
        active: Array,
    ) -> EstimatorState:
        def inactive_branch(_: None) -> EstimatorState:
            return estimator_state

        def active_branch(_: None) -> EstimatorState:
            sample_count = estimator_state.count + 1
            weight = 1.0 / sample_count.astype(sample_marginal.dtype)
            return EstimatorState(
                marginal=estimator_state.marginal + weight * (sample_marginal - estimator_state.marginal),
                marginal_dx=estimator_state.marginal_dx + weight * (sample_marginal_dx - estimator_state.marginal_dx),
                mean_force_num=estimator_state.mean_force_num + weight * (sample_mean_force_num - estimator_state.mean_force_num),
                count=sample_count,
            )

        return jax.lax.cond(active, active_branch, inactive_branch, operand=None)

    def step_fn(state: SimulationState) -> SimulationState:
        sample_marginal, sample_marginal_dx, sample_mean_force_num = kernel_observables(
            state.positions,
            grid,
            potential,
            estimator,
        )
        current_time = state.step.astype(jnp.float32) * config.dt
        estimator_state = update_estimator(
            state.estimator,
            sample_marginal,
            sample_marginal_dx,
            sample_mean_force_num,
            current_time >= estimator.burn_in,
        )
        bias_grid, _ = bias_from_estimator(
            estimator_state,
            method,
            config.alpha,
            gamma,
            potential.beta,
            estimator.floor,
        )
        bias_x = interpolation_from_uniform_grid(grid, bias_grid, state.positions[:, 0])
        drift = -entropic_potential_gradient(state.positions, potential)
        drift = drift.at[:, 0].add(bias_x)

        key, noise_key = jax.random.split(state.key)
        noise = noise_scale * jax.random.normal(noise_key, state.positions.shape)
        new_positions = state.positions + config.dt * drift + noise
        return SimulationState(
            positions=new_positions,
            estimator=estimator_state,
            key=key,
            step=state.step + 1,
        )

    return jax.jit(step_fn), grid, gamma


def make_observe_fn(config: ClassicalMDConfig):
    method = config.method.lower()
    potential = config.potential
    estimator = config.estimator
    grid = make_x_grid(potential, estimator)
    gamma = resolve_gamma(config.alpha, config.gamma)

    def observe_fn(state: SimulationState) -> Observation:
        sample_marginal, _, sample_mean_force_num = kernel_observables(
            state.positions,
            grid,
            potential,
            estimator,
        )
        safe_sample_marginal = jnp.maximum(sample_marginal, estimator.floor)
        instant_mean_force = sample_mean_force_num / safe_sample_marginal
        bias_grid, mean_force = bias_from_estimator(
            state.estimator,
            method,
            config.alpha,
            gamma,
            potential.beta,
            estimator.floor,
        )
        return Observation(
            positions=state.positions,
            marginal=state.estimator.marginal,
            mean_force=mean_force,
            bias=bias_grid,
            instant_marginal=sample_marginal,
            instant_mean_force=instant_mean_force,
        )

    return jax.jit(observe_fn)


def summarize_observation(
    observation: Observation,
    capture_particles: bool,
    store_estimators: bool,
) -> dict[str, np.ndarray]:
    positions = np.asarray(jax.device_get(observation.positions))
    marginal = np.asarray(jax.device_get(observation.marginal))
    mean_force = np.asarray(jax.device_get(observation.mean_force))
    bias = np.asarray(jax.device_get(observation.bias))
    instant_marginal = np.asarray(jax.device_get(observation.instant_marginal))
    instant_mean_force = np.asarray(jax.device_get(observation.instant_mean_force))

    summary = {
        "marginal_mean": marginal.mean(axis=0),
        "marginal_var": marginal.var(axis=0),
        "mean_force_mean": mean_force.mean(axis=0),
        "mean_force_var": mean_force.var(axis=0),
        "bias_mean": bias.mean(axis=0),
        "bias_var": bias.var(axis=0),
        "instant_marginal_mean": instant_marginal.mean(axis=0),
        "instant_marginal_var": instant_marginal.var(axis=0),
        "instant_mean_force_mean": instant_mean_force.mean(axis=0),
        "instant_mean_force_var": instant_mean_force.var(axis=0),
    }
    if capture_particles:
        summary["particles"] = positions
    if store_estimators:
        summary["marginal_estimates"] = marginal
        summary["mean_force_estimates"] = mean_force
        summary["bias_estimates"] = bias
    return summary


def run_simulation(
    config: ClassicalMDConfig,
    capture_particles: bool | None = None,
) -> SimulationResult:
    if config.method.lower() not in {"abf", "abp"}:
        raise ValueError(f"Unsupported method: {config.method}")
    if config.method.lower() == "abp" and not math.isfinite(config.alpha):
        raise ValueError("ABP requires a finite alpha.")
    if config.method.lower() == "abp" and config.gamma is not None:
        raise ValueError("ABP does not use gamma; pass alpha only.")
    if config.save_stride <= 0:
        raise ValueError("save_stride must be positive.")
    if capture_particles is None:
        capture_particles = config.store_particles

    step_fn, grid, gamma = make_step_fn(config)
    observe_fn = make_observe_fn(config)
    state = make_initial_state(config)

    times: list[float] = [0.0]
    summaries = [summarize_observation(observe_fn(state), capture_particles, config.store_estimators)]

    for step in range(1, config.num_steps + 1):
        state = step_fn(state)
        if step % config.save_stride == 0 or step == config.num_steps:
            times.append(step * config.dt)
            summaries.append(
                summarize_observation(
                    observe_fn(state),
                    capture_particles,
                    config.store_estimators,
                )
            )

    def stack(field: str) -> np.ndarray | None:
        values = [summary[field] for summary in summaries if field in summary]
        if not values:
            return None
        return np.stack(values, axis=0)

    reference_free_energy_values, reference_free_energy_gradient = reference_free_energy_and_gradient(grid, config.potential)
    x_grid = np.asarray(jax.device_get(grid))
    return SimulationResult(
        method=config.method.lower(),
        alpha=config.alpha,
        gamma=gamma,
        x_grid=x_grid,
        times=np.asarray(times),
        free_energy=np.asarray(jax.device_get(reference_free_energy_values)),
        free_energy_gradient=np.asarray(jax.device_get(reference_free_energy_gradient)),
        marginal_mean=stack("marginal_mean"),
        marginal_var=stack("marginal_var"),
        mean_force_mean=stack("mean_force_mean"),
        mean_force_var=stack("mean_force_var"),
        bias_mean=stack("bias_mean"),
        bias_var=stack("bias_var"),
        instant_marginal_mean=stack("instant_marginal_mean"),
        instant_marginal_var=stack("instant_marginal_var"),
        instant_mean_force_mean=stack("instant_mean_force_mean"),
        instant_mean_force_var=stack("instant_mean_force_var"),
        particles=stack("particles"),
        marginal_estimates=stack("marginal_estimates"),
        mean_force_estimates=stack("mean_force_estimates"),
        bias_estimates=stack("bias_estimates"),
    )


def result_to_dict(result: SimulationResult, config: ClassicalMDConfig) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "method": np.asarray(result.method),
        "alpha": np.asarray(result.alpha),
        "gamma": np.asarray(result.gamma),
        "times": result.times,
        "x_grid": result.x_grid,
        "free_energy": result.free_energy,
        "free_energy_gradient": result.free_energy_gradient,
        "marginal_mean": result.marginal_mean,
        "marginal_var": result.marginal_var,
        "mean_force_mean": result.mean_force_mean,
        "mean_force_var": result.mean_force_var,
        "bias_mean": result.bias_mean,
        "bias_var": result.bias_var,
        "instant_marginal_mean": result.instant_marginal_mean,
        "instant_marginal_var": result.instant_marginal_var,
        "instant_mean_force_mean": result.instant_mean_force_mean,
        "instant_mean_force_var": result.instant_mean_force_var,
        "config_json": np.asarray(json.dumps(asdict(config), sort_keys=True)),
    }
    if config.store_particles and result.particles is not None:
        payload["particles"] = result.particles
    if config.store_estimators:
        optional_fields = {
            "marginal_estimates": result.marginal_estimates,
            "mean_force_estimates": result.mean_force_estimates,
            "bias_estimates": result.bias_estimates,
        }
        for key, value in optional_fields.items():
            if value is not None:
                payload[key] = value
    return payload


def default_output_path(config: ClassicalMDConfig) -> Path:
    alpha_str = "inf" if math.isinf(config.alpha) else f"{config.alpha:g}"
    gamma = resolve_gamma(config.alpha, config.gamma)
    gamma_str = f"{gamma:g}"
    theta_str = f"{config.potential.theta:g}"
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / (
        f"{config.method.lower()}_alpha{alpha_str}_gamma{gamma_str}_"
        f"d{config.potential.d:g}_theta{theta_str}.npz"
    )


def save_result(result: SimulationResult, config: ClassicalMDConfig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result_to_dict(result, config))


def select_particle_indices(num_particles: int, max_particles: int) -> np.ndarray:
    if max_particles <= 0 or max_particles >= num_particles:
        return np.arange(num_particles, dtype=int)
    return np.unique(np.linspace(0, num_particles - 1, max_particles, dtype=int))


def make_trajectory_gif(
    result: SimulationResult,
    config: ClassicalMDConfig,
    output_path: Path,
    fps: int = 12,
    frame_stride: int = 1,
    max_particles: int = 32,
    trail_length: int = 8,
    grid_size: int = 180,
) -> Path:
    if result.particles is None:
        raise ValueError("Trajectory GIF generation requires captured particle snapshots.")
    if fps <= 0:
        raise ValueError("gif fps must be positive.")
    if frame_stride <= 0:
        raise ValueError("gif frame stride must be positive.")
    if trail_length < 0:
        raise ValueError("gif trail length must be non-negative.")
    if grid_size < 32:
        raise ValueError("gif grid size must be at least 32.")

    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    particles = np.asarray(result.particles)
    frame_indices = np.arange(0, particles.shape[0], frame_stride, dtype=int)
    if frame_indices[-1] != particles.shape[0] - 1:
        frame_indices = np.append(frame_indices, particles.shape[0] - 1)
    selected = select_particle_indices(particles.shape[1], max_particles)
    particle_frames = particles[frame_indices][:, selected, :]
    frame_times = result.times[frame_indices]

    flattened = particle_frames.reshape(-1, 2)
    x_extent = max(config.potential.xmax, float(np.max(np.abs(flattened[:, 0]))) + 0.35)
    y_extent = max(1.75, float(np.max(np.abs(flattened[:, 1]))) + 0.35)

    x_grid = np.linspace(-x_extent, x_extent, grid_size)
    y_grid = np.linspace(-y_extent, y_extent, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="xy")
    mesh_points = jnp.asarray(np.stack((xx, yy), axis=-1))
    potential_values = np.asarray(jax.device_get(entropic_potential(mesh_points, config.potential)))
    contour_levels = np.quantile(potential_values, np.linspace(0.1, 0.9, 9))
    contour_levels = np.unique(contour_levels)

    fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    ax.contourf(xx, yy, potential_values, levels=32, cmap="YlGnBu", alpha=0.35)
    if contour_levels.size >= 2:
        ax.contour(xx, yy, potential_values, levels=contour_levels, colors="#6b7a88", linewidths=0.8, alpha=0.85)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
    if abs(config.potential.theta) > ROTATION_TOL:
        channel_extent = max(x_extent, y_extent)
        channel_axis = np.array([-channel_extent, channel_extent])
        ax.plot(
            channel_axis * math.cos(config.potential.theta),
            channel_axis * math.sin(config.potential.theta),
            linestyle="--",
            linewidth=1.3,
            color="#8b1e3f",
            alpha=0.9,
        )

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, selected.size))
    trail_lines = [
        ax.plot([], [], color=colors[idx], linewidth=1.0, alpha=0.5)[0]
        for idx in range(selected.size)
    ]
    initial_positions = particle_frames[0]
    scatter = ax.scatter(
        initial_positions[:, 0],
        initial_positions[:, 1],
        s=30,
        c=colors,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )
    title = ax.set_title("")
    ax.set_xlim(-x_extent, x_extent)
    ax.set_ylim(-y_extent, y_extent)
    ax.set_aspect("equal")
    ax.set_xlabel("x (reaction coordinate used by the estimator)")
    ax.set_ylabel("y")

    def update(frame_id: int):
        current = particle_frames[frame_id]
        scatter.set_offsets(current)
        if trail_length == 0:
            for line in trail_lines:
                line.set_data([], [])
        else:
            start = max(0, frame_id - trail_length + 1)
            trail = particle_frames[start : frame_id + 1]
            for particle_id, line in enumerate(trail_lines):
                line.set_data(trail[:, particle_id, 0], trail[:, particle_id, 1])
        title.set_text(
            f"{config.method.upper()}  t={frame_times[frame_id]:.3f}  "
            f"theta={config.potential.theta:.3f} rad"
        )
        return [scatter, title, *trail_lines]

    animation = FuncAnimation(fig, update, frames=particle_frames.shape[0], interval=1000 / fps, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)
    return output_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JAX classical-MD ABF/ABP simulator on the entropic potential.")
    parser.add_argument("--method", choices=("abf", "abp"), default="abf")
    parser.add_argument("--num-trajectories", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=10_000)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-stride", type=int, default=100)
    parser.add_argument("--store-particles", action="store_true")
    parser.add_argument("--store-estimators", action="store_true")

    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--kappa0", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--d", type=float, default=10.0)
    parser.add_argument("--xmax", type=float, default=2.5)
    parser.add_argument("--theta", type=float, default=0.0, help="Counter-clockwise rotation angle of the potential in radians.")

    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--bandwidth", type=float, default=0.1)
    parser.add_argument("--floor", type=float, default=1e-8)
    parser.add_argument("--burn-in", type=float, default=0.0)
    parser.add_argument("--init-x", type=float, default=-1.0)
    parser.add_argument("--init-y", type=float, default=0.0)
    parser.add_argument("--init-std", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--make-gif", action="store_true", help="Generate a trajectory GIF from the saved particle snapshots.")
    parser.add_argument("--gif-output", type=Path, default=None)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--gif-frame-stride", type=int, default=1)
    parser.add_argument("--gif-max-particles", type=int, default=32)
    parser.add_argument("--gif-trail-length", type=int, default=8)
    parser.add_argument("--gif-grid-size", type=int, default=180)
    return parser


def config_from_args(args: argparse.Namespace) -> ClassicalMDConfig:
    return ClassicalMDConfig(
        method=args.method,
        num_trajectories=args.num_trajectories,
        num_steps=args.num_steps,
        dt=args.dt,
        alpha=args.alpha,
        gamma=args.gamma,
        seed=args.seed,
        save_stride=args.save_stride,
        store_particles=args.store_particles,
        store_estimators=args.store_estimators,
        potential=EntropicPotentialConfig(
            beta=args.beta,
            kappa0=args.kappa0,
            sigma=args.sigma,
            d=args.d,
            xmax=args.xmax,
            theta=args.theta * (math.pi / 180.0),  # Convert degrees to radians if needed,
        ),
        estimator=EstimatorConfig(
            grid_size=args.grid_size,
            bandwidth=args.bandwidth,
            floor=args.floor,
            burn_in=args.burn_in,
        ),
        initial=InitialConditionConfig(
            center_x=args.init_x,
            center_y=args.init_y,
            std=args.init_std,
        ),
    )


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    result = run_simulation(config, capture_particles=(config.store_particles or args.make_gif))
    output_path = args.output or default_output_path(config)
    save_result(result, config, output_path)
    gif_path: Path | None = None
    if args.make_gif:
        gif_path = args.gif_output or output_path.with_suffix(".gif")
        make_trajectory_gif(
            result,
            config,
            gif_path,
            fps=args.gif_fps,
            frame_stride=args.gif_frame_stride,
            max_particles=args.gif_max_particles,
            trail_length=args.gif_trail_length,
            grid_size=args.gif_grid_size,
        )
    print(
        f"Saved {config.method.lower()} run with {config.num_trajectories} trajectories, "
        f"T={config.num_steps * config.dt:g}, alpha={config.alpha:g}, gamma={result.gamma:g}, "
        f"theta={config.potential.theta:g} "
        f"to {output_path}"
    )
    if gif_path is not None:
        print(f"Saved trajectory GIF to {gif_path}")


if __name__ == "__main__":
    main()
