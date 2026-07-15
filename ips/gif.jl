########################################################################
# 2D gate potential: U(x,y) = (x^2-1)^2 + (1+kappa*exp(-x^2/(2*sigma^2)))^(d-1)*y^2
# kappa=1.0, sigma=0.25, beta=1.0, d is a runtime parameter (main() sweeps
# d in {2, 10}, same as ../mean_field/gif.jl -- no more const d / d==2
# special case).
#
# Finite-particle counterpart of ../mean_field/gif.jl: SAME potential,
# SAME panel layout/timing/alpha grid/initial condition -- but here ABF
# and ABP estimate their own bias ON-LINE from the N-particle ensemble
# (running mean-force bin-average for ABF, Gaussian-KDE marginal for ABP,
# exactly as in l1_decay.jl), rather than reading the precomputed FreeFEM
# bias fields ../mean_field/gif.jl uses. This is the genuine interacting
# particle system (the IPS/ tier from the top-level README); any visible
# difference from ../mean_field/gif.jl's animation at the same N and
# alphas is finite-population estimation noise, not a different model.
#
# For each d in {2, 10}, animates 8 panels (2x4 grid), matching
# ../mean_field/gif.jl's layout exactly:
#   row 1: ABP(alpha=0.125) | ABP(alpha=1) | ABP(alpha=4) | ABP(alpha=16)
#   row 2: Unbiased         | ABF(alpha=1) | ABF(alpha=4) | ABF(alpha=16)
# ABF's bias uses gamma=alpha/(1+alpha) at alpha=1,4,16 -- the SAME alpha
# values ABP uses -- so panels are a matched-alpha comparison of the two
# algorithms.
#
# Zoomed in on the EARLY transient only: simulated up to SIM_TMAX=0.5,
# densely sampled (same dt, frame schedule and fps as
# ../mean_field/gif.jl), so the GIF shows the initial pile-up breaking
# apart rather than convergence to steady state.
#
# Backend: Plots.jl + GR (gr()) -- the standard backend for animations/
# GIF export (unlike plotlyjs(), which targets interactive HTML).
#
# Produces: particles_d2.gif, particles_d10.gif
########################################################################

using Random, Statistics
using Plots
gr()

# ---------------------------------------------------------------------
# 2D potential and partial derivatives (d is a runtime parameter, as in
# ../mean_field/gif.jl).
# ---------------------------------------------------------------------
const KAPPA = 1.0
const SIGMA = 0.25

g(x) = exp(-x^2 / (2 * SIGMA^2))

U(x, y, d)    = (x^2 - 1)^2 + (1 + KAPPA * g(x))^(d - 1) * y^2
dUdx(x, y, d) = 4x * (x^2 - 1) - (d - 1) * (KAPPA * x / SIGMA^2) * g(x) * (1 + KAPPA * g(x))^(d - 2) * y^2
dUdy(x, y, d) = 2 * (1 + KAPPA * g(x))^(d - 1) * y

# ---------------------------------------------------------------------
# Hyperparameters -- domain, initial condition and frame schedule match
# ../mean_field/gif.jl exactly; sigma_kernel/nbins/eps_reg are specific to
# this file's on-line bias estimators (no mean_field equivalent).
# ---------------------------------------------------------------------
Base.@kwdef mutable struct Params
    N::Int              = 3000
    dt::Float64          = 5e-4
    nsteps::Int          = round(Int, 0.5 / 5e-4)
    beta::Float64         = 1.0
    d::Int              = 2
    sigma_kernel::Float64  = 0.02   # Gaussian kernel width used inside ABP's bias construction
    eps_reg::Float64       = 1e-3
    x0_mean::Float64      = -1.0
    y0_mean::Float64      = 0.0
    init_std::Float64     = 0.05
    seed::Int            = 1

    xmin::Float64 = -2.5
    xmax::Float64 = 2.5
    ymin::Float64 = -2.5
    ymax::Float64 = 2.5
    nbins::Int   = 60       # bins over x only, for ABF's running mean-force average

    # frame-saving schedule -- identical to ../mean_field/gif.jl
    early_phase_steps::Int = 400
    save_every_early::Int  = 4
    save_every::Int       = 32
    fps::Int             = 100
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

function init_particles(p::Params, rng)
    x = p.x0_mean .+ p.init_std .* randn(rng, p.N)
    y = p.y0_mean .+ p.init_std .* randn(rng, p.N)
    x .= reflect.(x, p.xmin, p.xmax)
    y .= reflect.(y, p.ymin, p.ymax)
    return x, y
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
    x, y = init_particles(p, rng)

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
            fx = dUdx(xi, yi, p.d); fy = dUdy(xi, yi, p.d)
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
    x, y = init_particles(p, rng)

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
            floc = dUdx(xi, yi, p.d)
            sumF[i] += floc; cnt[i] += 1
            Aprime[i] = sumF[i] / cnt[i]

            bias = gamma * Aprime[i]
            fy = dUdy(xi, yi, p.d)

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
    x, y = init_particles(p, rng)

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
            fx = dUdx(xi, yi, p.d); fy = dUdy(xi, yi, p.d)

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
# Build the 2x4 animated GIF (8 method panels, exactly filling the grid):
# potential contour in background, particle scatter on top -- same
# layout function as ../mean_field/gif.jl's build_particle_gif.
# ---------------------------------------------------------------------
function build_particle_gif(results, labels, p::Params, outname::String;
                              ngrid::Int = 120, clim_max::Float64 = 3.0)
    xg = range(p.xmin, p.xmax, length = ngrid)
    yg = range(p.ymin, p.ymax, length = ngrid)
    Z  = [U(xi, yi, p.d) for yi in yg, xi in xg]   # Z[iy,ix] = U(xg[ix], yg[iy], d)

    nframes = size(results[1][1], 2)   # all panels share the same frame count/schedule

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
        plot(panels..., layout = (2, 4), size = (2200, 1000),
             plot_title = "Finite-particle dynamics (N=$(p.N), d=$(p.d), κ=$(KAPPA), σ=$(SIGMA), β=$(p.beta))")
    end

    gif(anim, outname, fps = p.fps)
end

const ALPHA_VLOW = 0.125  # ABP only
const ALPHA_LOW  = 1.0    # shared by ABP and ABF
const ALPHA_MID  = 4.0    # shared by ABP and ABF
const ALPHA_HIGH = 16.0   # shared by ABP and ABF

function run_d(d::Int)
    p = Params(d = d, seed = 1)

    println("d=$(d): running Unbiased...")
    r_unb = run_traj_unbiased(p)

    println("d=$(d): running ABP(alpha=$(ALPHA_VLOW))...")
    r_abp_vlow = run_traj_ABP(p, ALPHA_VLOW)
    println("d=$(d): running ABP(alpha=$(ALPHA_LOW))...")
    r_abp_low = run_traj_ABP(p, ALPHA_LOW)
    println("d=$(d): running ABP(alpha=$(ALPHA_MID))...")
    r_abp_mid = run_traj_ABP(p, ALPHA_MID)
    println("d=$(d): running ABP(alpha=$(ALPHA_HIGH))...")
    r_abp_high = run_traj_ABP(p, ALPHA_HIGH)

    println("d=$(d): running ABF(alpha=$(ALPHA_LOW))...")
    r_abf_low = run_traj_ABF(p, ALPHA_LOW / (1 + ALPHA_LOW))
    println("d=$(d): running ABF(alpha=$(ALPHA_MID))...")
    r_abf_mid = run_traj_ABF(p, ALPHA_MID / (1 + ALPHA_MID))
    println("d=$(d): running ABF(alpha=$(ALPHA_HIGH))...")
    r_abf_high = run_traj_ABF(p, ALPHA_HIGH / (1 + ALPHA_HIGH))

    outname = "particles_d$(d).gif"
    println("d=$(d): building $(outname)...")
    # row 1: ABP(vlow) | ABP(low) | ABP(mid) | ABP(high)
    # row 2: Unbiased  | ABF(low) | ABF(mid) | ABF(high)
    build_particle_gif((r_abp_vlow, r_abp_low, r_abp_mid, r_abp_high,
                        r_unb, r_abf_low, r_abf_mid, r_abf_high),
             ("ABP (α=$(ALPHA_VLOW))", "ABP (α=$(ALPHA_LOW))", "ABP (α=$(ALPHA_MID))", "ABP (α=$(ALPHA_HIGH))",
              "Unbiased", "ABF (α=$(ALPHA_LOW))", "ABF (α=$(ALPHA_MID))", "ABF (α=$(ALPHA_HIGH))"),
             p, outname)
    println("Saved $(outname)")
end

function main()
    for d in (2, 10)
        run_d(d)
    end
end

main()
