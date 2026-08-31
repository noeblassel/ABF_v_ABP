########################################################################
# 2D PURELY ENTROPIC two-well geometry (sharper version of gif.jl's gate
# potential): the potential is FLAT (U == 0) everywhere inside an
# L-shaped-in-y domain
#
#     Omega = B((-1,0), 0.8)  U  B((+1,0), 0.8)  U  tube
#     tube  = { |x| <= 1,  |y| <= eps/2 }
#
# i.e. two circular wells joined by a thin straight tube of width `eps`.
# Particles feel NO force anywhere; the boundary of Omega is a hard wall
# (reflecting -- implemented by axis-wise sliding / rejection). All of
# the free-energy barrier along the reaction coordinate xi(x,y) = x is
# therefore ENTROPIC: it comes entirely from the collapse of the
# accessible y-length from ~1.6 inside a well down to `eps` inside the
# tube,  A(x) = -beta^-1 log(len_y(x)).  Making `eps` smaller makes the
# barrier sharper -- `eps` is the swept parameter here, exactly as `d`
# is in gif.jl / ../mean_field/gif.jl.
#
# Finite-particle counterpart of gif.jl: SAME panel layout / timing /
# alpha grid / initial condition, and ABF and ABP again estimate their
# own bias ON-LINE from the N-particle ensemble (running mean-force
# bin-average for ABF, Gaussian-KDE marginal for ABP). The one
# structural difference forced by the flat potential:
#
#   * ABP biases with the empirical marginal density rho(x) directly, so
#     it DOES see the entropic bottleneck and pushes particles through
#     the tube.
#   * ABF's mean-force sample is f = dU/dx == 0 everywhere, so its
#     running average stays 0 and ABF reproduces the unbiased dynamics.
#     A force-only ABF is blind to a purely entropic barrier (the
#     missing piece is the geometric/boundary term, not in dU/dx). The
#     ABF row is kept so the contrast with the ABP row is visible.
#
# ABF TIME RESCALING ("updatedtime" variant): ABF(alpha) integrates
# with an effective step dt_eff = (1 + alpha) * dt (both drift and
# noise), and its panels report this rescaled "updated time". ABF's
# converged reaction-coordinate drift carries a factor 1 - gamma =
# 1/(1+alpha); multiplying the step by (1 + alpha) cancels it, so
# frame-for-frame ABF(alpha) is comparable to ABP(alpha) rather than
# (1+alpha)x slower. The ABF panel clock therefore runs ahead of the
# ABP/Unbiased clock by the same factor.
#
# For each eps in EPS_VALUES, animates 8 panels (2x4 grid):
#   row 1: ABP(alpha=0.125) | ABP(alpha=1) | ABP(alpha=4) | ABP(alpha=16)
#   row 2: Unbiased         | ABF(alpha=1) | ABF(alpha=4) | ABF(alpha=16)
#
# Backend: Plots.jl + GR (gr()).
# Produces: entropic_particles_eps<eps>.gif  (one per swept eps)
########################################################################

using Random, Statistics
using Plots
gr()

# ---------------------------------------------------------------------
# Geometry of the flat domain Omega (see header). `eps` (tube width) is
# a runtime parameter, passed through Params like `d` in gif.jl.
# ---------------------------------------------------------------------
const XL    = -1.0   # left well centre x
const XR    =  1.0   # right well centre x
const RWELL =  0.8   # well radius  (B(-1,0.8), B(1,0.8))
const XTUBE =  1.0   # tube spans |x| <= XTUBE

in_left(x, y)       = (x - XL)^2 + y^2 <= RWELL^2
in_right(x, y)      = (x - XR)^2 + y^2 <= RWELL^2
in_tube(x, y, eps)  = abs(x) <= XTUBE && abs(y) <= eps / 2
in_domain(x, y, eps) = in_left(x, y) || in_right(x, y) || in_tube(x, y, eps)

# Flat potential: no force anywhere. Kept as functions so the ABF/ABP
# integrators below stay byte-for-byte parallel to gif.jl.
U(x, y, eps)    = 0.0
dUdx(x, y, eps) = 0.0
dUdy(x, y, eps) = 0.0

# ---------------------------------------------------------------------
# Hyperparameters -- initial condition and frame schedule mirror
# gif.jl; SIM_TMAX is longer here because escape through the tube is
# slow. sigma_kernel/nbins/eps_reg belong to the on-line bias
# estimators.
# ---------------------------------------------------------------------
const SIM_TMAX = 1.0

Base.@kwdef mutable struct Params
    N::Int              = 3000
    dt::Float64          = 5e-4
    nsteps::Int          = round(Int, SIM_TMAX / 5e-4)
    beta::Float64         = 1.0
    eps::Float64          = 0.1    # tube width -- the swept parameter
    sigma_kernel::Float64  = 0.2   # Gaussian kernel width inside ABP's bias construction
    eps_reg::Float64       = 1e-3
    x0_mean::Float64      = -1.0
    y0_mean::Float64      = 0.0
    init_std::Float64     = 0.05
    seed::Int            = 1

    xmin::Float64 = -2.0
    xmax::Float64 = 2.0
    ymin::Float64 = -1.0
    ymax::Float64 = 1.0
    nbins::Int   = 60       # bins over x only, for ABF's running mean-force average

    # frame-saving schedule -- identical structure to gif.jl
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

# Hard-wall move against the domain boundary: try the full 2D step; if
# it leaves Omega, try sliding along x only, then along y only; if all
# fail, stay put. (Rejection/sliding reflection -- robust for the curved
# well boundary and the thin tube without needing the wall normal.)
@inline function step_domain(xo, yo, dx, dy, p::Params)
    xp = reflect(xo + dx, p.xmin, p.xmax)
    yp = reflect(yo + dy, p.ymin, p.ymax)
    in_domain(xp, yp, p.eps) && return (xp, yp)
    in_domain(xp, yo, p.eps) && return (xp, yo)
    in_domain(xo, yp, p.eps) && return (xo, yp)
    return (xo, yo)
end

function init_particles(p::Params, rng)
    x = zeros(p.N); y = zeros(p.N)
    for k in 1:p.N
        while true
            xk = p.x0_mean + p.init_std * randn(rng)
            yk = p.y0_mean + p.init_std * randn(rng)
            if in_domain(xk, yk, p.eps)
                x[k] = xk; y[k] = yk
                break
            end
        end
    end
    return x, y
end

function save_schedule(p::Params)
    early = collect(1:p.save_every_early:min(p.early_phase_steps, p.nsteps))
    late_start = last(early) + p.save_every
    late = collect(late_start:p.save_every:p.nsteps)
    return vcat(early, late)
end

# ---------------------------------------------------------------------
# UNBIASED: reflected Brownian motion in Omega (flat potential)
# ---------------------------------------------------------------------
function run_traj_unbiased(p::Params)
    rng = MersenneTwister(p.seed)
    x, y = init_particles(p, rng)

    schedule = save_schedule(p)
    traj_x = zeros(p.N, length(schedule) + 1)
    traj_y = zeros(p.N, length(schedule) + 1)
    times  = zeros(length(schedule) + 1)
    traj_x[:, 1] .= x; traj_y[:, 1] .= y

    sig = sqrt(2 * p.dt / p.beta)
    frame = 1; nxt = 1
    for step in 1:p.nsteps
        t = step * p.dt
        for k in 1:p.N
            dx = -dUdx(x[k], y[k], p.eps) * p.dt + sig * randn(rng)
            dy = -dUdy(x[k], y[k], p.eps) * p.dt + sig * randn(rng)
            x[k], y[k] = step_domain(x[k], y[k], dx, dy, p)
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
# ABF: adaptive biasing FORCE, applied only to x, with gamma =
# alpha/(1+alpha). "updatedtime" variant: the Euler-Maruyama step is
# rescaled to dt_eff = (1 + alpha) * dt (drift AND noise), and `times`
# reports this rescaled clock, so ABF(alpha) is frame-comparable to
# ABP(alpha) (see header). f-sample = dU/dx == 0 here, so Aprime stays
# 0 and the biased force vanishes -- only the time rescaling survives.
# ---------------------------------------------------------------------
function run_traj_ABF(p::Params, alpha::Float64)
    gamma = alpha / (1 + alpha)
    dt_eff = (1 + alpha) * p.dt
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

    sig = sqrt(2 * dt_eff / p.beta)
    frame = 1; nxt = 1
    for step in 1:p.nsteps
        t = step * dt_eff
        for k in 1:p.N
            xi, yi = x[k], y[k]
            i    = binidx(xi)
            floc = dUdx(xi, yi, p.eps)
            sumF[i] += floc; cnt[i] += 1
            Aprime[i] = sumF[i] / cnt[i]

            bias = gamma * Aprime[i]
            dx = (-floc + bias) * dt_eff + sig * randn(rng)
            dy = -dUdy(xi, yi, p.eps) * dt_eff + sig * randn(rng)
            x[k], y[k] = step_domain(xi, yi, dx, dy, p)
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
# ABP: adaptive biasing POTENTIAL, applied only to x, fixed alpha.
# Uses the ensemble's empirical marginal rho(x) -- so it DOES see the
# entropic bottleneck.
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

    sig = sqrt(2 * p.dt / p.beta)
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
            dx = (-dUdx(xi, yi, p.eps) + bias) * p.dt + sig * randn(rng)
            dy = -dUdy(xi, yi, p.eps) * p.dt + sig * randn(rng)
            x[k], y[k] = step_domain(xi, yi, dx, dy, p)
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
# Build the 2x4 animated GIF: domain mask in the background (light =
# inside Omega), particle scatter on top. Same layout as gif.jl's
# build_particle_gif.
# ---------------------------------------------------------------------
function build_particle_gif(results, labels, p::Params, outname::String; ngrid::Int = 240)
    xg = range(p.xmin, p.xmax, length = ngrid)
    yg = range(p.ymin, p.ymax, length = ngrid)
    Z  = [in_domain(xi, yi, p.eps) ? 1.0 : 0.0 for yi in yg, xi in xg]

    nframes = size(results[1][1], 2)

    anim = @animate for f in 1:nframes
        panels = Plots.Plot[]
        for (idx, (traj_x, traj_y, times)) in enumerate(results)
            sp = heatmap(xg, yg, Z, color = cgrad([:white, :lightsteelblue]),
                         colorbar = false, xlabel = "x", ylabel = "y",
                         xlim = (p.xmin, p.xmax), ylim = (p.ymin, p.ymax),
                         aspect_ratio = :equal,
                         title = "$(labels[idx])   t=$(round(times[f], digits = 2))",
                         titlefontsize = 10)
            scatter!(sp, traj_x[:, f], traj_y[:, f], color = :red, ms = 2.5,
                     markerstrokewidth = 0, label = false)
            push!(panels, sp)
        end
        plot(panels..., layout = (2, 4), size = (2200, 1000),
             plot_title = "Entropic two-well (N=$(p.N), ε=$(p.eps), β=$(p.beta)) — flat potential, reflecting walls")
    end

    gif(anim, outname, fps = p.fps)
end

const ALPHA_VLOW = 0.125  # ABP only
const ALPHA_LOW  = 1.0    # shared by ABP and ABF
const ALPHA_MID  = 4.0    # shared by ABP and ABF
const ALPHA_HIGH = 16.0   # shared by ABP and ABF

const EPS_VALUES = (0.1, 0.01)   # tube widths to sweep (sharper = smaller)

function run_eps(eps::Float64)
    p = Params(eps = eps, seed = 1)

    println("eps=$(eps): running Unbiased...")
    r_unb = run_traj_unbiased(p)

    println("eps=$(eps): running ABP(alpha=$(ALPHA_VLOW))...")
    r_abp_vlow = run_traj_ABP(p, ALPHA_VLOW)
    println("eps=$(eps): running ABP(alpha=$(ALPHA_LOW))...")
    r_abp_low = run_traj_ABP(p, ALPHA_LOW)
    println("eps=$(eps): running ABP(alpha=$(ALPHA_MID))...")
    r_abp_mid = run_traj_ABP(p, ALPHA_MID)
    println("eps=$(eps): running ABP(alpha=$(ALPHA_HIGH))...")
    r_abp_high = run_traj_ABP(p, ALPHA_HIGH)

    println("eps=$(eps): running ABF(alpha=$(ALPHA_LOW))...")
    r_abf_low = run_traj_ABF(p, ALPHA_LOW)
    println("eps=$(eps): running ABF(alpha=$(ALPHA_MID))...")
    r_abf_mid = run_traj_ABF(p, ALPHA_MID)
    println("eps=$(eps): running ABF(alpha=$(ALPHA_HIGH))...")
    r_abf_high = run_traj_ABF(p, ALPHA_HIGH)

    outname = "entropic_particles_hightube_eps$(eps).gif"
    println("eps=$(eps): building $(outname)...")
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
    for eps in EPS_VALUES
        run_eps(Float64(eps))
    end
end

main()
