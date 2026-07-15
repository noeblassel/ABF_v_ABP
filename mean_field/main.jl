########################################################################
# 2D potential:
#   U(x,y) = (x^2 - 1)^2 + (1 + kappa*exp(-x^2/(2*sigma^2)))^(d-1) * y^2
#
#   x: slow/reaction coordinate (double well, barrier at x=0)
#   y: transverse "gate" coordinate -- tightly confined near x=0
#      (stiffness ~ kappa), essentially unconfined near the wells
#      x=+-1 since g(x)=exp(-x^2/(2 sigma^2)) is tiny there (sigma=0.25).
#
# Parameters: sigma=0.25, kappa=1.0, beta=1.0.
#
# MEAN-FIELD BIAS (this file's difference from multithreaded2daveraged.jl):
# instead of building the bias from the particles' own empirical
# histogram/KDE (self-consistent, on-line estimate), the bias applied
# to each particle at (x,t) is read directly from precomputed files
# supplied by a collaborator, one per (algorithm, alpha, d):
#
#   biases/abp_alpha<alpha>_d<d>_bias.txt
#   biases/abf_alpha<alpha>_d<d>_bias.txt
#
# Each file holds 1001 lines (t = 0:0.005:5) x 151 columns (x on a
# uniform grid from xmin=-2.5 to xmax=2.5). Each entry is ALREADY the
# fully-formed bias force b(x,t) for that (algorithm, alpha, d) --
# i.e. exactly the term that gets added to the drift, alpha/gamma
# already folded in (verified numerically: the ABP row at t=0 is
# antisymmetric about the initial condition's mean x=-1 and scales
# exactly linearly with alpha across files). No further rescaling is
# applied here -- the file value is used as-is.
#
# The bias field is bilinearly interpolated: linearly in x between
# the 151 grid columns, and linearly in t between the 1001 time rows
# (the particle simulation uses a finer dt than the file's 0.005).
#
# Unbiased runs use no file at all (plain Langevin in the same U(x,y;d)).
#
# METRIC (replacing the TV/L1-vs-target-density metric used previously):
# for every particle we record the first-passage time to x=0 (tau0)
# and the first-passage time to x=1 (tau1); by continuity of the path
# tau1 >= tau0 always (x0=-1 < 0 < 1). We report, versus time, the
# cumulative fraction of particles with tau0<=t ("passed 0") and with
# tau1<=t ("passed 0 then passed 1").
#
# Particles are initialised i.i.d. Gaussian: x0 ~ N(-1, 0.05^2),
# y0 ~ N(0, 0.05^2) -- matching the initial condition implicit in the
# supplied bias files.
#
# Produces ONE output per dimension in DIMS_TO_RUN:
#   meanfield_crossing_fractions_d<d>.html
#   -- 3 panels (Unbiased | ABP(alpha) | ABF(alpha)); solid = fraction
#      that passed 0, dashed = fraction that passed 0 then 1, colored
#      by alpha. Curves are averages over N_REPS independent replicates.
#
# Parallelism: Threads.@threads over the flattened (alpha, rep) job
# list, same pattern as multithreaded2daveraged.jl. Launch with
#   julia --threads=32 main.jl
########################################################################

using Random, Statistics, DelimitedFiles
using Plots
plotlyjs()

# ---------------------------------------------------------------------
# 2D potential and partial derivatives (d now a runtime parameter)
# ---------------------------------------------------------------------
const KAPPA = 1.0
const SIGMA = 0.25

g(x) = exp(-x^2 / (2 * SIGMA^2))

U(x, y, d)    = (x^2 - 1)^2 + (1 + KAPPA * g(x))^(d - 1) * y^2
dUdx(x, y, d) = 4x * (x^2 - 1) - (d - 1) * (KAPPA * x / SIGMA^2) * g(x) * (1 + KAPPA * g(x))^(d - 2) * y^2
dUdy(x, y, d) = 2 * (1 + KAPPA * g(x))^(d - 1) * y

# ---------------------------------------------------------------------
# Precomputed mean-field bias field: loaded from disk, bilinearly
# interpolated in (x, t).
# ---------------------------------------------------------------------
const BIAS_DIR   = joinpath(@__DIR__, "biases")
const GRID_XMIN  = -2.5
const GRID_XMAX  = 2.5
const GRID_NX    = 151
const FILE_DT    = 0.005
const FILE_NT    = 1001
const TFINAL     = 5.0

# alpha value -> exact string used in the filenames
const ALPHA_STRINGS = Dict(
    0.125 => "0.125", 0.25 => "0.25", 0.5 => "0.5", 1.0 => "1",
    2.0 => "2", 4.0 => "4", 8.0 => "8", 16.0 => "16",
)
const ALPHAS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

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
    N::Int          = 300
    dt::Float64      = 5e-4
    nsteps::Int      = round(Int, TFINAL / 5e-4)
    beta::Float64     = 1.0
    alpha::Float64    = 1.0
    d::Int          = 2
    x0_mean::Float64  = -1.0
    y0_mean::Float64  = 0.0
    init_std::Float64 = 0.05
    seed::Int        = 1

    xmin::Float64 = GRID_XMIN
    xmax::Float64 = GRID_XMAX
    ymin::Float64 = -2.5
    ymax::Float64 = 2.5
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

# first-passage times to x=0 and (subsequently) x=1
function crossing_fractions(fpt0::Vector{Float64}, fpt1::Vector{Float64}, tgrid::Vector{Float64})
    n = length(fpt0)
    frac0 = [count(<=(t), fpt0) / n for t in tgrid]
    frac1 = [count(<=(t), fpt1) / n for t in tgrid]
    return frac0, frac1
end

# ---------------------------------------------------------------------
# UNBIASED: plain 2D overdamped Langevin (no bias in x or y)
# ---------------------------------------------------------------------
function run_unbiased(p::Params)
    rng = MersenneTwister(p.seed)
    x, y = init_particles(p, rng)

    fpt0 = fill(NaN, p.N)
    fpt1 = fill(NaN, p.N)
    escaped0 = falses(p.N)
    escaped1 = falses(p.N)

    for step in 1:p.nsteps
        t = step * p.dt
        for k in 1:p.N
            xi, yi = x[k], y[k]
            fx = dUdx(xi, yi, p.d)
            fy = dUdy(xi, yi, p.d)
            x[k] = reflect(xi + (-fx) * p.dt + sqrt(2 * p.dt / p.beta) * randn(rng), p.xmin, p.xmax)
            y[k] = reflect(yi + (-fy) * p.dt + sqrt(2 * p.dt / p.beta) * randn(rng), p.ymin, p.ymax)

            if !escaped0[k] && x[k] >= 0
                escaped0[k] = true
                fpt0[k] = t
            end
            if !escaped1[k] && x[k] >= 1
                escaped1[k] = true
                fpt1[k] = t
            end
        end
    end

    return fpt0, fpt1
end

# ---------------------------------------------------------------------
# ABP: bias force read directly from the precomputed mean-field file
# (applied to x only; y always unbiased).
# ---------------------------------------------------------------------
function run_ABP(p::Params, bf::BiasField)
    rng = MersenneTwister(p.seed)
    x, y = init_particles(p, rng)

    fpt0 = fill(NaN, p.N)
    fpt1 = fill(NaN, p.N)
    escaped0 = falses(p.N)
    escaped1 = falses(p.N)

    for step in 1:p.nsteps
        t = step * p.dt
        for k in 1:p.N
            xi, yi = x[k], y[k]
            fx = dUdx(xi, yi, p.d)
            fy = dUdy(xi, yi, p.d)
            bias = bias_at(bf, xi, t)

            x[k] = reflect(xi + (-fx - bias) * p.dt + sqrt(2 * p.dt / p.beta) * randn(rng), p.xmin, p.xmax)
            y[k] = reflect(yi + (-fy) * p.dt        + sqrt(2 * p.dt / p.beta) * randn(rng), p.ymin, p.ymax)

            if !escaped0[k] && x[k] >= 0
                escaped0[k] = true
                fpt0[k] = t
            end
            if !escaped1[k] && x[k] >= 1
                escaped1[k] = true
                fpt1[k] = t
            end
        end
    end

    return fpt0, fpt1
end

# ---------------------------------------------------------------------
# ABF: bias force read directly from the precomputed mean-field file
# (applied to x only; y always unbiased).
# ---------------------------------------------------------------------
function run_ABF(p::Params, bf::BiasField)
    rng = MersenneTwister(p.seed)
    x, y = init_particles(p, rng)

    fpt0 = fill(NaN, p.N)
    fpt1 = fill(NaN, p.N)
    escaped0 = falses(p.N)
    escaped1 = falses(p.N)

    for step in 1:p.nsteps
        t = step * p.dt
        for k in 1:p.N
            xi, yi = x[k], y[k]
            fx = dUdx(xi, yi, p.d)
            fy = dUdy(xi, yi, p.d)
            bias = bias_at(bf, xi, t)

            x[k] = reflect(xi + (-fx - bias) * p.dt + sqrt(2 * p.dt / p.beta) * randn(rng), p.xmin, p.xmax)
            y[k] = reflect(yi + (-fy) * p.dt        + sqrt(2 * p.dt / p.beta) * randn(rng), p.ymin, p.ymax)

            if !escaped0[k] && x[k] >= 0
                escaped0[k] = true
                fpt0[k] = t
            end
            if !escaped1[k] && x[k] >= 1
                escaped1[k] = true
                fpt1[k] = t
            end
        end
    end

    return fpt0, fpt1
end

# =======================================================================
# Parallel alpha sweep (for one dimension d), WITH REPLICATES, using the
# precomputed mean-field bias files. All (alpha, rep) jobs for ABP+ABF,
# plus all rep jobs for the unbiased reference, are flattened into ONE
# job list dispatched via a single Threads.@threads loop.
# =======================================================================
function run_alpha_sweep(d::Int; n_reps::Int = 10, N::Int = 300, dt::Float64 = 5e-4,
                            beta::Float64 = 1.0, ymin::Float64 = -2.5, ymax::Float64 = 2.5,
                            tgrid::Vector{Float64} = collect(0:0.01:TFINAL), base_seed::Int = 1000)

    n = length(ALPHAS)
    nsteps = round(Int, TFINAL / dt)
    println("d=$(d): alpha sweep over $(n) values: $(ALPHAS), n_reps=$(n_reps).")
    println("Running with $(Threads.nthreads()) threads.")

    bias_abp = Dict(a => load_bias_field("abp", a, d) for a in ALPHAS)
    bias_abf = Dict(a => load_bias_field("abf", a, d) for a in ALPHAS)

    frac0_abp_reps = [Vector{Vector{Float64}}(undef, n_reps) for _ in 1:n]
    frac1_abp_reps = [Vector{Vector{Float64}}(undef, n_reps) for _ in 1:n]
    frac0_abf_reps = [Vector{Vector{Float64}}(undef, n_reps) for _ in 1:n]
    frac1_abf_reps = [Vector{Vector{Float64}}(undef, n_reps) for _ in 1:n]

    frac0_unb_reps = Vector{Vector{Float64}}(undef, n_reps)
    frac1_unb_reps = Vector{Vector{Float64}}(undef, n_reps)

    jobs = Tuple{Symbol,Int,Int}[]
    for i in 1:n, r in 1:n_reps
        push!(jobs, (:sweep, i, r))
    end
    for r in 1:n_reps
        push!(jobs, (:unbiased, 0, r))
    end
    njobs = length(jobs)
    println("Total parallel jobs: $(njobs)")

    Threads.@threads for jidx in 1:njobs
        kind, i, r = jobs[jidx]

        if kind == :sweep
            a = ALPHAS[i]
            seed_abp = base_seed + 2 * ((i - 1) * n_reps + r)
            seed_abf = seed_abp + 1

            p_abp = Params(N = N, dt = dt, nsteps = nsteps, beta = beta, alpha = a, d = d,
                            seed = seed_abp, ymin = ymin, ymax = ymax)
            p_abf = Params(N = N, dt = dt, nsteps = nsteps, beta = beta, alpha = a, d = d,
                            seed = seed_abf, ymin = ymin, ymax = ymax)

            fpt0_abp, fpt1_abp = run_ABP(p_abp, bias_abp[a])
            fpt0_abf, fpt1_abf = run_ABF(p_abf, bias_abf[a])

            f0a, f1a = crossing_fractions(fpt0_abp, fpt1_abp, tgrid)
            f0f, f1f = crossing_fractions(fpt0_abf, fpt1_abf, tgrid)

            frac0_abp_reps[i][r] = f0a; frac1_abp_reps[i][r] = f1a
            frac0_abf_reps[i][r] = f0f; frac1_abf_reps[i][r] = f1f

            println("  d=$(d) alpha=$(a) rep=$(r)/$(n_reps) done (thread $(Threads.threadid()))  final: ABP(0,1)=$(f0a[end]),$(f1a[end])  ABF(0,1)=$(f0f[end]),$(f1f[end])")

        else # :unbiased
            seed = base_seed + 1_000_000 + r
            p_unb = Params(N = N, dt = dt, nsteps = nsteps, beta = beta, d = d,
                            seed = seed, ymin = ymin, ymax = ymax)
            fpt0_unb, fpt1_unb = run_unbiased(p_unb)
            f0u, f1u = crossing_fractions(fpt0_unb, fpt1_unb, tgrid)
            frac0_unb_reps[r] = f0u; frac1_unb_reps[r] = f1u
            println("  d=$(d) Unbiased rep=$(r)/$(n_reps) done (thread $(Threads.threadid()))  final: (0,1)=$(f0u[end]),$(f1u[end])")
        end
    end

    avg_curve(vecs) = mean(hcat(vecs...), dims = 2)[:, 1]

    frac0_abp = [avg_curve(frac0_abp_reps[i]) for i in 1:n]
    frac1_abp = [avg_curve(frac1_abp_reps[i]) for i in 1:n]
    frac0_abf = [avg_curve(frac0_abf_reps[i]) for i in 1:n]
    frac1_abf = [avg_curve(frac1_abf_reps[i]) for i in 1:n]
    frac0_unb = avg_curve(frac0_unb_reps)
    frac1_unb = avg_curve(frac1_unb_reps)

    return (d = d, ALPHAS = ALPHAS, n_reps = n_reps, tgrid = tgrid,
            frac0_abp = frac0_abp, frac1_abp = frac1_abp,
            frac0_abf = frac0_abf, frac1_abf = frac1_abf,
            frac0_unb = frac0_unb, frac1_unb = frac1_unb)
end

# ---------------------------------------------------------------------
# Figure: 3 panels (Unbiased | ABP(alpha) | ABF(alpha)); solid = frac
# passed 0, dashed = frac passed 0 then 1, colored by alpha.
# ---------------------------------------------------------------------
function build_sweep_crossing_figure(res)
    ALPHAS = res.ALPHAS
    n = length(ALPHAS)
    logA = log10.(ALPHAS)
    lo, hi = extrema(logA)
    colors = [cgrad(:viridis)[(logA[i] - lo) / max(hi - lo, eps())] for i in 1:n]

    p0 = plot(title = "Unbiased", xlabel = "t", ylabel = "cumulative fraction",
              ylim = (0, 1), legend = :outertopright, legendfontsize = 6)
    plot!(p0, res.tgrid, res.frac0_unb, color = :gray, lw = 2, ls = :solid,
          label = "passed 0 ($(round(Int, 100*res.frac0_unb[end]))%)")
    plot!(p0, res.tgrid, res.frac1_unb, color = :gray, lw = 2, ls = :dash,
          label = "then passed 1 ($(round(Int, 100*res.frac1_unb[end]))%)")

    p1 = plot(title = "ABP(alpha)", xlabel = "t", ylabel = "cumulative fraction",
              ylim = (0, 1), legend = :outertopright, legendfontsize = 6)
    p2 = plot(title = "ABF(alpha)", xlabel = "t", ylabel = "cumulative fraction",
              ylim = (0, 1), legend = :outertopright, legendfontsize = 6)

    for i in 1:n
        plot!(p1, res.tgrid, res.frac0_abp[i], color = colors[i], lw = 1.5, ls = :solid,
              label = "α=$(ALPHAS[i]) 0→($(round(Int, 100*res.frac0_abp[i][end]))%)")
        plot!(p1, res.tgrid, res.frac1_abp[i], color = colors[i], lw = 1.5, ls = :dash, label = "")
        plot!(p2, res.tgrid, res.frac0_abf[i], color = colors[i], lw = 1.5, ls = :solid,
              label = "α=$(ALPHAS[i]) 0→($(round(Int, 100*res.frac0_abf[i][end]))%)")
        plot!(p2, res.tgrid, res.frac1_abf[i], color = colors[i], lw = 1.5, ls = :dash, label = "")
    end

    plot!(p1, res.tgrid, res.frac0_unb, color = :black, lw = 2, ls = :solid, label = "Unbiased 0")
    plot!(p1, res.tgrid, res.frac1_unb, color = :black, lw = 2, ls = :dash, label = "Unbiased 1")
    plot!(p2, res.tgrid, res.frac0_unb, color = :black, lw = 2, ls = :solid, label = "Unbiased 0")
    plot!(p2, res.tgrid, res.frac1_unb, color = :black, lw = 2, ls = :dash, label = "Unbiased 1")

    plot(p0, p1, p2, layout = (1, 3), size = (1800, 600),
         plot_title = "Mean-field bias — crossing fractions vs time (d=$(res.d), κ=$(KAPPA), σ=$(SIGMA), β=1.0, avg of $(res.n_reps) reps)")
end

# DIMS_TO_RUN controls which of the collaborator's precomputed dimensions
# (2..10) get simulated; extend/shorten this list as needed.
const DIMS_TO_RUN = [8]

function main_sweep()
    N_REPS = 10

    for d in DIMS_TO_RUN
        res = run_alpha_sweep(d; n_reps = N_REPS, N = 300, dt = 5e-4,
                                 beta = 1.0, ymin = -2.5, ymax = 2.5,
                                 tgrid = collect(0:0.01:TFINAL), base_seed = 800)

        fig = build_sweep_crossing_figure(res)
        outname = "meanfield_crossing_fractions_d$(d).html"
        savefig(fig, outname)
        println("Saved $(outname)")
    end
end

main_sweep()
