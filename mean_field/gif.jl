########################################################################
# Mean-field particle animation: the SAME idealized dynamics as main.jl
# (bias read from the precomputed FreeFEM bias fields in biases/, applied
# identically to N otherwise-independent replicas) -- NOT the interacting
# particle system in ../finite_particles/, which instead estimates its own
# bias on-line from the ensemble. Potential, bias loader/interpolator and
# integrators are the trajectory-recording counterparts of main.jl's
# run_unbiased/run_ABP/run_ABF (first-passage times only there; here the
# full (x,y) path is stored for the animation).
#
# Zoomed in on the EARLY transient only: simulated up to SIM_TMAX (rather
# than the bias files' full TFINAL=5.0 domain), densely sampled, so the
# GIF shows the initial pile-up breaking apart rather than the long tail
# of already-separated particles drifting to steady state.
#
# For each d in {2, 10}, animates 8 panels (2x4 grid):
#   row 1: ABP(alpha=0.125) | ABP(alpha=1) | ABP(alpha=4) | ABP(alpha=16)
#   row 2: Unbiased         | ABF(alpha=1) | ABF(alpha=4) | ABF(alpha=16)
# ABP sweeps all four alphas; ABF shares the alpha=1/4/16 triple with ABP
# (a matched-alpha comparison of the two algorithms), with Unbiased taking
# the remaining slot in row 2.
#
# All 8 panels for a given d reuse ONE Params (one seed, reseeded per
# method call): the noise realization is identical across panels, so the
# only visible difference between panels is the bias itself.
#
# Parameters: kappa0=1.0, sigma=0.25, beta=1.0 (same as main.jl / edp/).
#
# Backend: Plots.jl + GR (gr()) -- as in finite_particles/gif.jl, the
# standard backend for GIF export (unlike plotlyjs(), used for the
# interactive HTML in main.jl).
#
# Produces: meanfield_particles_d2.gif, meanfield_particles_d10.gif
########################################################################

using Random, Statistics, DelimitedFiles
using Plots
gr()

# ---------------------------------------------------------------------
# 2D potential and partial derivatives (d is a runtime parameter) --
# identical to main.jl.
# ---------------------------------------------------------------------
const KAPPA = 1.0
const SIGMA = 0.25

g(x) = exp(-x^2 / (2 * SIGMA^2))

U(x, y, d)    = (x^2 - 1)^2 + (1 + KAPPA * g(x))^(d - 1) * y^2
dUdx(x, y, d) = 4x * (x^2 - 1) - (d - 1) * (KAPPA * x / SIGMA^2) * g(x) * (1 + KAPPA * g(x))^(d - 2) * y^2
dUdy(x, y, d) = 2 * (1 + KAPPA * g(x))^(d - 1) * y

# ---------------------------------------------------------------------
# Precomputed mean-field bias field: loaded from disk, bilinearly
# interpolated in (x, t) -- identical to main.jl.
# ---------------------------------------------------------------------
const BIAS_DIR   = joinpath(@__DIR__, "biases")
const GRID_XMIN  = -2.5
const GRID_XMAX  = 2.5
const GRID_NX    = 151
const FILE_DT    = 0.005
const FILE_NT    = 1001
const TFINAL     = 5.0   # time domain covered by the bias files (used to clamp bias_at)
const SIM_TMAX   = 0.5   # this script only simulates/animates the early part of [0, TFINAL]

# alpha value -> exact string used in the filenames
const ALPHA_STRINGS = Dict(
    0.125 => "0.125", 0.25 => "0.25", 0.5 => "0.5", 1.0 => "1",
    2.0 => "2", 4.0 => "4", 8.0 => "8", 16.0 => "16",
)

struct BiasField
    data::Matrix{Float64}   # (FILE_NT, GRID_NX), rows=t, cols=x
end

function load_bias_field(algorithm::String, alpha::Float64, d::Int)
    fname = joinpath(BIAS_DIR, "$(algorithm)_alpha$(ALPHA_STRINGS[alpha])_d$(d)_bias.txt")
    data = readdlm(fname, Float64)
    @assert size(data) == (FILE_NT, GRID_NX) "unexpected shape $(size(data)) for $fname"
    return BiasField(data)
end

@inline function bias_at(bf::BiasField, x::Float64, t::Float64)
    tc = clamp(t, 0.0, TFINAL)
    ft = tc / FILE_DT
    i0 = clamp(floor(Int, ft), 0, FILE_NT - 1)
    i1 = min(i0 + 1, FILE_NT - 1)
    wt = ft - i0

    xc = clamp(x, GRID_XMIN, GRID_XMAX)
    dxg = (GRID_XMAX - GRID_XMIN) / (GRID_NX - 1)
    fx = (xc - GRID_XMIN) / dxg
    j0 = clamp(floor(Int, fx), 0, GRID_NX - 2)
    j1 = j0 + 1
    wx = fx - j0

    @inbounds begin
        v00 = bf.data[i0 + 1, j0 + 1]; v01 = bf.data[i0 + 1, j1 + 1]
        v10 = bf.data[i1 + 1, j0 + 1]; v11 = bf.data[i1 + 1, j1 + 1]
    end
    v0 = v00 * (1 - wx) + v01 * wx
    v1 = v10 * (1 - wx) + v11 * wx
    return v0 * (1 - wt) + v1 * wt
end

# ---------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------
Base.@kwdef mutable struct Params
    N::Int          = 500
    dt::Float64      = 5e-4
    nsteps::Int      = round(Int, SIM_TMAX / 5e-4)
    beta::Float64     = 1.0
    d::Int          = 2
    x0_mean::Float64  = -1.0
    y0_mean::Float64  = 0.0
    init_std::Float64 = 0.05
    seed::Int        = 1

    xmin::Float64 = GRID_XMIN
    xmax::Float64 = GRID_XMAX
    ymin::Float64 = -2.5
    ymax::Float64 = 2.5

    # frame-saving schedule: dense early (to catch the initial jump),
    # a bit sparser later -- same idea as finite_particles/gif.jl, but
    # rescaled to the short SIM_TMAX=0.5 window so the whole animation
    # stays densely sampled (300 frames total instead of the ~95 used
    # for the old full-T=5 run).
    early_phase_steps::Int = 400
    save_every_early::Int  = 8
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
# UNBIASED: plain 2D overdamped Langevin (no bias in x or y)
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
            fx = dUdx(xi, yi, p.d)
            fy = dUdy(xi, yi, p.d)
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
# ABP / ABF: bias force read directly from the precomputed mean-field
# file (applied to x only; y always unbiased). Identical code for both
# methods -- only the loaded BiasField (abp vs abf file) differs, exactly
# as in main.jl's run_ABP/run_ABF.
# ---------------------------------------------------------------------
function run_traj_biased(p::Params, bf::BiasField)
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
            fx = dUdx(xi, yi, p.d)
            fy = dUdy(xi, yi, p.d)
            bias = bias_at(bf, xi, t)

            x[k] = reflect(xi + (-fx - bias) * p.dt + sqrt(2 * p.dt / p.beta) * randn(rng), p.xmin, p.xmax)
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
# potential contour in background, particle scatter on top.
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
             plot_title = "Mean-field dynamics (d=$(p.d), κ=$(KAPPA), σ=$(SIGMA), β=$(p.beta))")
    end

    gif(anim, outname, fps = p.fps)
end

const ALPHA_VLOW = 0.125  # ABP only
const ALPHA_LOW  = 1.0    # shared by ABP and ABF
const ALPHA_MID  = 4.0    # shared by ABP and ABF
const ALPHA_HIGH = 16.0   # shared by ABP and ABF

function run_d(d::Int)
    p = Params(d = d, seed = 1)

    println("d=$(d): loading bias fields (alpha=$(ALPHA_VLOW),$(ALPHA_LOW),$(ALPHA_MID),$(ALPHA_HIGH))...")
    bf_abp_vlow = load_bias_field("abp", ALPHA_VLOW, d)
    bf_abp_low  = load_bias_field("abp", ALPHA_LOW,  d)
    bf_abp_mid  = load_bias_field("abp", ALPHA_MID,  d)
    bf_abp_high = load_bias_field("abp", ALPHA_HIGH, d)
    bf_abf_low  = load_bias_field("abf", ALPHA_LOW,  d)
    bf_abf_mid  = load_bias_field("abf", ALPHA_MID,  d)
    bf_abf_high = load_bias_field("abf", ALPHA_HIGH, d)

    println("d=$(d): running Unbiased...")
    r_unb = run_traj_unbiased(p)
    println("d=$(d): running ABP(alpha=$(ALPHA_VLOW))...")
    r_abp_vlow = run_traj_biased(p, bf_abp_vlow)
    println("d=$(d): running ABP(alpha=$(ALPHA_LOW))...")
    r_abp_low = run_traj_biased(p, bf_abp_low)
    println("d=$(d): running ABP(alpha=$(ALPHA_MID))...")
    r_abp_mid = run_traj_biased(p, bf_abp_mid)
    println("d=$(d): running ABP(alpha=$(ALPHA_HIGH))...")
    r_abp_high = run_traj_biased(p, bf_abp_high)
    println("d=$(d): running ABF(alpha=$(ALPHA_LOW))...")
    r_abf_low = run_traj_biased(p, bf_abf_low)
    println("d=$(d): running ABF(alpha=$(ALPHA_MID))...")
    r_abf_mid = run_traj_biased(p, bf_abf_mid)
    println("d=$(d): running ABF(alpha=$(ALPHA_HIGH))...")
    r_abf_high = run_traj_biased(p, bf_abf_high)

    outname = "meanfield_particles_d$(d).gif"
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
