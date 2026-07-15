########################################################################
# 2D gate potential: U(x,y) = (x^2-1)^2 + (1+kappa*exp(-x^2/(2*sigma^2)))^(d-1)*y^2
# kappa=1.0, sigma=0.25, beta=1.0, d=5.
#
# GIF of particle positions moving in the (x,y) plane, potential shown
# as a filled contour in the background, 4 panels side by side:
#   1) Unbiased overdamped Langevin
#   2) ABF, gamma = 0.5   (i.e. alpha=1, gamma=alpha/(1+alpha))
#   3) ABP, alpha = 1
#   4) ABP, alpha = 10
#
# Deliberately FEW steps, so the GIF shows the initial condition and
# the transient regime (particles starting piled up at x0=-1) rather
# than full convergence to steady state.
#
# Backend: Plots.jl + GR (gr()) -- the standard backend for animations/
# GIF export (unlike plotlyjs(), which targets interactive HTML).
########################################################################

using Random, Statistics
using Plots
gr()

# ---------------------------------------------------------------------
# 2D potential and partial derivatives
# ---------------------------------------------------------------------
const KAPPA = 1.0
const SIGMA = 0.25
const d = 5

g(x)  = exp(-x^2 / (2 * SIGMA^2))

U(x, y)    = (x^2 - 1)^2 + (1 + KAPPA * g(x)) ^ (d-1) * y^2
if d == 2
    dUdx(x, y) = 4x * (x^2 - 1) -  (KAPPA * x / SIGMA^2) * g(x) * y^2
else
    dUdx(x, y) = 4x * (x^2 - 1) - (d-1) * (KAPPA * x / SIGMA^2) * g(x) * (1 + KAPPA * g(x)) ^ (d-2) * y^2
end
dUdy(x, y) = 2 * (1 + KAPPA * g(x)) ^ (d-1)* y

# ---------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------
Base.@kwdef mutable struct Params
    N::Int              = 150      # number of particles shown
    dt::Float64          = 1e-3
    nsteps::Int          = 4_000    # FEW steps: initial condition + transient only
    beta::Float64         = 1.0
    sigma_kernel::Float64  = 0.02   # Gaussian kernel width used inside ABP's bias construction
    eps_reg::Float64       = 1e-3
    x0::Float64           = -1.0
    y0::Float64           = 0.0
    seed::Int            = 1

    xmin::Float64 = -2.5
    xmax::Float64 = 2.5
    ymin::Float64 = -2.5
    ymax::Float64 = 2.5
    nbins::Int   = 60       # bins over x only, for ABF's running mean-force average

    # frame-saving schedule: dense early (to catch the initial jump),
    # sparse later
    early_phase_steps::Int = 800
    save_every_early::Int  = 15
    save_every::Int       = 60
    fps::Int             = 12
end

@inline function reflect(xi, xmin, xmax)
    if xi < xmin
        return xmin + (xmin - xi)
    elseif xi > xmax
        return xmax - (xi - xmax)
    else
        return xi
    end
end

function save_schedule(p::Params)
    early = collect(1:p.save_every_early:min(p.early_phase_steps, p.nsteps))
    late_start = last(early) + p.save_every
    late = collect(late_start:p.save_every:p.nsteps)
    return vcat(early, late)
end

# ---------------------------------------------------------------------
# UNBIASED: plain 2D overdamped Langevin
# ---------------------------------------------------------------------
function run_traj_unbiased(p::Params)
    rng = MersenneTwister(p.seed)
    x = fill(p.x0, p.N); y = fill(p.y0, p.N)

    schedule = save_schedule(p)
    traj_x = zeros(p.N, length(schedule) + 1)
    traj_y = zeros(p.N, length(schedule) + 1)
    times  = zeros(length(schedule) + 1)
    traj_x[:, 1] .= x; traj_y[:, 1] .= y

    frame = 1; nxt = 1
    for step in 1:p.nsteps
        t = step * p.dt
        for k in 1:p.N
            xi, yi = x[k], y[k]
            fx = dUdx(xi, yi); fy = dUdy(xi, yi)
            x[k] = reflect(xi + (-fx) * p.dt + sqrt(2 * p.dt / p.beta) * randn(rng), p.xmin, p.xmax)
            y[k] = reflect(yi + (-fy) * p.dt + sqrt(2 * p.dt / p.beta) * randn(rng), p.ymin, p.ymax)
        end
        if nxt <= length(schedule) && step == schedule[nxt]
            frame += 1
            traj_x[:, frame] .= x; traj_y[:, frame] .= y
            times[frame] = t
            nxt += 1
        end
    end
    return traj_x, traj_y, times
end

# ---------------------------------------------------------------------
# ABF: adaptive biasing FORCE, applied only to x, fixed gamma
# ---------------------------------------------------------------------
function run_traj_ABF(p::Params, gamma::Float64)
    rng = MersenneTwister(p.seed)
    x = fill(p.x0, p.N); y = fill(p.y0, p.N)

    edges = range(p.xmin, p.xmax, length = p.nbins + 1)
    dxbin = step(edges)
    sumF   = zeros(p.nbins)
    cnt    = zeros(p.nbins)
    Aprime = zeros(p.nbins)
    binidx(xi) = clamp(Int(floor((xi - p.xmin) / dxbin)) + 1, 1, p.nbins)

    schedule = save_schedule(p)
    traj_x = zeros(p.N, length(schedule) + 1)
    traj_y = zeros(p.N, length(schedule) + 1)
    times  = zeros(length(schedule) + 1)
    traj_x[:, 1] .= x; traj_y[:, 1] .= y

    frame = 1; nxt = 1
    for step in 1:p.nsteps
        t = step * p.dt
        for k in 1:p.N
            xi, yi = x[k], y[k]
            i    = binidx(xi)
            floc = dUdx(xi, yi)
            sumF[i] += floc; cnt[i] += 1
            Aprime[i] = sumF[i] / cnt[i]

            bias = gamma * Aprime[i]
            fy = dUdy(xi, yi)

            x[k] = reflect(xi + (-floc + bias) * p.dt + sqrt(2 * p.dt / p.beta) * randn(rng), p.xmin, p.xmax)
            y[k] = reflect(yi + (-fy) * p.dt          + sqrt(2 * p.dt / p.beta) * randn(rng), p.ymin, p.ymax)
        end
        if nxt <= length(schedule) && step == schedule[nxt]
            frame += 1
            traj_x[:, frame] .= x; traj_y[:, frame] .= y
            times[frame] = t
            nxt += 1
        end
    end
    return traj_x, traj_y, times
end

# ---------------------------------------------------------------------
# ABP: adaptive biasing POTENTIAL, applied only to x, fixed alpha
# ---------------------------------------------------------------------
function run_traj_ABP(p::Params, alpha::Float64)
    rng = MersenneTwister(p.seed)
    x = fill(p.x0, p.N); y = fill(p.y0, p.N)

    edges   = range(p.xmin, p.xmax, length = p.nbins + 1)
    centers = collect((edges[1:end-1] .+ edges[2:end]) ./ 2)
    dxbin   = step(edges)

    B     = zeros(p.nbins)
    Bgrad = zeros(p.nbins)
    binidx(xi) = clamp(Int(floor((xi - p.xmin) / dxbin)) + 1, 1, p.nbins)
    gauss(u, s) = exp(-u^2 / (2s^2)) / (sqrt(2π) * s)

    schedule = save_schedule(p)
    traj_x = zeros(p.N, length(schedule) + 1)
    traj_y = zeros(p.N, length(schedule) + 1)
    times  = zeros(length(schedule) + 1)
    traj_x[:, 1] .= x; traj_y[:, 1] .= y

    frame = 1; nxt = 1
    for step in 1:p.nsteps
        t = step * p.dt
        rho = fill(1e-8, p.nbins)

        for k in 1:p.N
            xi = x[k]
            jlo = binidx(xi - 4p.sigma_kernel)
            jhi = binidx(xi + 4p.sigma_kernel)
            @inbounds for j in jlo:jhi
                rho[j] += gauss(centers[j] - xi, p.sigma_kernel) * p.dt
            end
        end

        s = sum(rho) * dxbin
        @inbounds for j in 1:p.nbins
            B[j] = -1 / p.beta * log(rho[j] / s + p.eps_reg)
        end
        @inbounds for j in 1:p.nbins
            jm, jp = max(j - 1, 1), min(j + 1, p.nbins)
            Bgrad[j] = (B[jp] - B[jm]) / ((jp - jm) * dxbin)
        end

        for k in 1:p.N
            xi, yi = x[k], y[k]
            i = binidx(xi)
            bias = alpha * Bgrad[i]
            fx = dUdx(xi, yi); fy = dUdy(xi, yi)

            x[k] = reflect(xi + (-fx + bias) * p.dt + sqrt(2 * p.dt / p.beta) * randn(rng), p.xmin, p.xmax)
            y[k] = reflect(yi + (-fy) * p.dt        + sqrt(2 * p.dt / p.beta) * randn(rng), p.ymin, p.ymax)
        end

        if nxt <= length(schedule) && step == schedule[nxt]
            frame += 1
            traj_x[:, frame] .= x; traj_y[:, frame] .= y
            times[frame] = t
            nxt += 1
        end
    end
    return traj_x, traj_y, times
end

# ---------------------------------------------------------------------
# Build the 2x2 animated GIF: potential contour in background,
# particle scatter on top, one panel per method.
# ---------------------------------------------------------------------
function make_gif(results, labels, p::Params; ngrid::Int = 120, clim_max::Float64 = 3.0)
    xg = range(p.xmin, p.xmax, length = ngrid)
    yg = range(p.ymin, p.ymax, length = ngrid)
    Z  = [U(xi, yi) for yi in yg, xi in xg]   # Z[iy,ix] = U(xg[ix], yg[iy])

    nframes = size(results[1][1], 2)   # all 4 share the same frame count/schedule

    anim = @animate for f in 1:nframes
        panels = Plots.Plot[]
        for (idx, (traj_x, traj_y, times)) in enumerate(results)
            sp = contourf(xg, yg, Z, color = :viridis, clims = (0, clim_max),
                          linewidth = 0, colorbar = false,
                          xlabel = "x", ylabel = "y",
                          xlim = (p.xmin, p.xmax), ylim = (p.ymin, p.ymax),
                          title = "$(labels[idx])   t=$(round(times[f], digits = 2))",
                          titlefontsize = 10)
            scatter!(sp, traj_x[:, f], traj_y[:, f], color = :red, ms = 2.5,
                     markerstrokewidth = 0, label = false)
            push!(panels, sp)
        end
        plot(panels..., layout = (2, 2), size = (1100, 1000))
    end

    gif(anim, "particles_2d.gif", fps = p.fps)
end

function main()
    p = Params(N = 300, dt = 1e-4, nsteps = 10_000, beta = 1.0,
               sigma_kernel = 0.02, x0 = -1.0, y0 = 0.0, seed = 1,
               xmin = -2.5, xmax = 2.5, ymin = -6.0, ymax = 6.0, nbins = 60,
               early_phase_steps = 800, save_every_early = 15, save_every = 60, fps = 12)

    println("Running Unbiased...")
    r_unb = run_traj_unbiased(p)

    println("Running ABF (gamma=0.5, i.e. alpha=1)...")
    r_abf = run_traj_ABF(p, 0.5)

    println("Running ABP (alpha=1)...")
    r_abp1 = run_traj_ABP(p, 1.0)

    println("Running ABP (alpha=10)...")
    r_abp10 = run_traj_ABP(p, 10.0)

    println("Building GIF...")
    make_gif((r_unb, r_abf, r_abp1, r_abp10),
             ("Unbiased", "ABF (γ=0.5)", "ABP (α=1)", "ABP (α=10)"), p)

    println("Saved particles_2d.gif")
end

main()