########################################################################
# 2D potential:
#   U(x,y) = (x^2 - 1)^2 + (1 + kappa*exp(-x^2/(2*sigma^2)))^(d-1) * y^2
#
#   x: slow/reaction coordinate (double well, barrier at x=0)
#   y: transverse "gate" coordinate -- tightly confined near x=0
#      (stiffness ~ kappa), essentially unconfined near the wells
#      x=+-1 since g(x)=exp(-x^2/(2 sigma^2)) is tiny there (sigma=0.25).
#
# Parameters: sigma=0.25, kappa=1.0, beta=1.0, d=2 (fixed in this file).
#
# Only x is biased/binned (xi(x,y)=x is the reaction coordinate); y
# always evolves under its own unbiased force. The ABF mean-force
# sample is f(x,y)=dU/dx(x,y); its running bin-average over visited
# (x,y) pairs converges to the correct conditional mean force
# E[dU/dx | x], exactly as in the general coarea/mean-force formula.
#
# The TRUE target marginal in x is NOT just (x^2-1)^2: integrating out
# y (Gaussian, x-dependent stiffness) adds an entropic correction. This
# is computed here numerically (Riemann sum over a y-grid on the SAME
# finite reflecting y-domain the dynamics uses), so the reference
# target is self-consistent with the truncated-box simulation rather
# than assuming an infinite y-domain.
#
# Produces ONE output: alpha_sweep_l1_vs_time_2d.html
#   -- L1(t) = ||rho_emp(t) - rho_infty||_1 curves (marginal in x),
#      one panel each for Unbiased / ABP(alpha) / ABF(gamma), escape
#      fraction (crossing x=0) shown directly in each curve's legend.
#
# REPLICATES: each (alpha) curve for ABP and ABF -- and the two fixed
# reference curves (Unbiased, ABF(gamma=1)) -- is now averaged over
# N_REPS independent replicate runs (different RNG seeds). All
# (alpha, replicate) jobs are flattened into one list and dispatched
# together across threads, so load is balanced evenly regardless of
# how many alphas vs. replicates there are.
#
# Plotting backend: Plots.jl + PlotlyJS (plotlyjs()) -- standard,
# reliable, self-contained interactive HTML. Nothing is displayed,
# only saved to disk.
#
# Parallelism: Threads.@threads over the flattened (alpha, rep) job
# list. Launch with
#   julia --threads=32 l1_decay.jl
########################################################################

using Random, Statistics
using Plots
plotlyjs()

# ---------------------------------------------------------------------
# 2D potential and partial derivatives
# ---------------------------------------------------------------------
const KAPPA = 1.0
const SIGMA = 0.25
const d = 2

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
    N::Int              = 300
    dt::Float64          = 1e-3
    nsteps::Int          = 20_000
    beta::Float64         = 1.0     # <-- as requested
    alpha::Float64        = 1.0
    sigma_kernel::Float64  = 0.02   # Gaussian kernel width used INSIDE ABP's bias construction (in x)
    eps_reg::Float64       = 1e-3
    x0::Float64           = -1.0
    y0::Float64           = 0.0
    seed::Int            = 1
    gamma::Float64         = 1.0     # ABF's own bias scaling factor

    xmin::Float64 = -2.5
    xmax::Float64 = 2.5
    nbins::Int   = 60       # bins over x only, for ABF's running mean-force average

    ymin::Float64 = -2.5
    ymax::Float64 = 2.5      # reflecting numerical box for the "gate" coordinate y
                              # (near the wells y is only loosely confined -- widen
                              #  this if you push beta/kappa/sigma/nsteps further)

    kde_bandwidth::Float64 = 0.05
    kde_grid_n::Int       = 200      # grid resolution (in x) for KDE / L1 integral
    ny_quad::Int          = 200      # grid resolution (in y) for numerically marginalizing U over y
    snapshot_every::Int   = 100
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

function build_grids(p::Params)
    xr = range(p.xmin, p.xmax, length = p.kde_grid_n)
    yr = range(p.ymin, p.ymax, length = p.ny_quad)
    return collect(xr), step(xr), collect(yr), step(yr)
end

# Marginal target density in x: rho_infty(x) propto integral_y exp(-beta*factor*U(x,y)) dy,
# integrated numerically (Riemann sum) over the SAME finite y-box the dynamics uses,
# then normalized (Riemann sum) over x.
function marginal_target_density(beta::Float64, factor::Float64,
                                   xgrid::Vector{Float64}, dxg::Float64,
                                   ygrid::Vector{Float64}, dyg::Float64)
    m = length(xgrid)
    raw = zeros(m)
    @inbounds for i in 1:m
        xi = xgrid[i]
        s = 0.0
        a = (xi^2 .- 1) .^ 2 - log(4 * pi / (beta * (1.0 + KAPPA * exp(- xi ^ 2 / (2 * SIGMA ^ 2))))) / (2 * beta)
        for yj in ygrid
            s += exp(-beta * U(xi, yj) + beta * factor * a)
        end
        raw[i] = s * dyg
    end
    Z = sum(raw) * dxg
    return raw ./ Z
end

# Gaussian-kernel KDE of the ensemble's x-coordinates on `grid`, bandwidth h
function kde_eval(xs::Vector{Float64}, grid::Vector{Float64}, h::Float64)
    n = length(xs)
    dens = zeros(length(grid))
    norm_const = 1.0 / (n * h * sqrt(2π))
    @inbounds for xi in xs
        for gi in eachindex(grid)
            u = (grid[gi] - xi) / h
            dens[gi] += exp(-0.5 * u^2)
        end
    end
    dens .*= norm_const
    return dens
end

l1_distance(rho_emp::Vector{Float64}, rho_inf::Vector{Float64}, dxg::Float64) =
    sum(abs.(rho_emp .- rho_inf)) * dxg

# ---------------------------------------------------------------------
# UNBIASED: plain 2D overdamped Langevin (no bias in x or y)
# ---------------------------------------------------------------------
function run_unbiased_fpt(p::Params)
    rng = MersenneTwister(p.seed)
    x = fill(p.x0, p.N)
    y = fill(p.y0, p.N)

    fpt     = fill(NaN, p.N)
    escaped = falses(p.N)

    xgrid, dxg, ygrid, dyg = build_grids(p)
    rho_inf  = marginal_target_density(p.beta, 0.0, xgrid, dxg, ygrid, dyg)
    l1_times = Float64[]
    l1_vals  = Float64[]
    push!(l1_times, 0.0)
    push!(l1_vals, l1_distance(kde_eval(x, xgrid, p.kde_bandwidth), rho_inf, dxg))

    for step in 1:p.nsteps
        t = step * p.dt
        for k in 1:p.N
            xi, yi = x[k], y[k]
            fx = dUdx(xi, yi)
            fy = dUdy(xi, yi)
            x[k] = reflect(xi + (-fx) * p.dt + sqrt(2 * p.dt / p.beta) * randn(rng), p.xmin, p.xmax)
            y[k] = reflect(yi + (-fy) * p.dt + sqrt(2 * p.dt / p.beta) * randn(rng), p.ymin, p.ymax)

            if !escaped[k] && x[k] > 0
                escaped[k] = true
                fpt[k] = t
            end
        end

        if step % p.snapshot_every == 0
            push!(l1_times, t)
            push!(l1_vals, l1_distance(kde_eval(x, xgrid, p.kde_bandwidth), rho_inf, dxg))
        end
    end

    return fpt, escaped, l1_times, l1_vals
end

# ---------------------------------------------------------------------
# ABF: adaptive biasing FORCE, applied only to x. gamma_override pins
# gamma to a fixed value (e.g. gamma=1) instead of alpha/(1+alpha).
# Mean-force sample f(x,y) = dU/dx(x,y); its running bin-average in x
# converges to the correct conditional mean force E[dU/dx | x].
# ---------------------------------------------------------------------
function run_ABF_fpt(p::Params; gamma_override::Union{Nothing,Float64} = nothing)
    p.gamma = gamma_override === nothing ? p.alpha / (1 + p.alpha) : gamma_override
    rng = MersenneTwister(p.seed)
    x = fill(p.x0, p.N)
    y = fill(p.y0, p.N)

    edges = range(p.xmin, p.xmax, length = p.nbins + 1)
    dxbin = step(edges)
    sumF   = zeros(p.nbins)
    cnt    = zeros(p.nbins)
    Aprime = zeros(p.nbins)
    binidx(xi) = clamp(Int(floor((xi - p.xmin) / dxbin)) + 1, 1, p.nbins)

    fpt     = fill(NaN, p.N)
    escaped = falses(p.N)

    xgrid, dxg, ygrid, dyg = build_grids(p)
    rho_inf  = marginal_target_density(p.beta, p.gamma, xgrid, dxg, ygrid, dyg)
    l1_times = Float64[]
    l1_vals  = Float64[]
    push!(l1_times, 0.0)
    push!(l1_vals, l1_distance(kde_eval(x, xgrid, p.kde_bandwidth), rho_inf, dxg))

    for step in 1:p.nsteps
        t = step * p.dt
        for k in 1:p.N
            xi, yi = x[k], y[k]
            i    = binidx(xi)
            floc = dUdx(xi, yi)
            sumF[i] += floc
            cnt[i]  += 1
            Aprime[i] = sumF[i] / cnt[i]

            bias = p.gamma * Aprime[i]
            fy = dUdy(xi, yi)

            x[k] = reflect(xi + (-floc + bias) * p.dt + sqrt(2 * p.dt / p.beta) * randn(rng), p.xmin, p.xmax)
            y[k] = reflect(yi + (-fy) * p.dt        + sqrt(2 * p.dt / p.beta) * randn(rng), p.ymin, p.ymax)

            if !escaped[k] && x[k] > 0
                escaped[k] = true
                fpt[k] = t
            end
        end

        if step % p.snapshot_every == 0
            push!(l1_times, t)
            push!(l1_vals, l1_distance(kde_eval(x, xgrid, p.kde_bandwidth), rho_inf, dxg))
        end
    end

    return fpt, escaped, l1_times, l1_vals
end

# ---------------------------------------------------------------------
# ABP: adaptive biasing POTENTIAL, applied only to x. Kernel-density
# estimate is built from the ensemble's x-coordinates only (y ignored
# for the bias, though still simulated/unbiased).
# ---------------------------------------------------------------------
function run_ABP_fpt(p::Params)
    rng = MersenneTwister(p.seed)
    x = fill(p.x0, p.N)
    y = fill(p.y0, p.N)

    edges   = range(p.xmin, p.xmax, length = p.nbins + 1)
    centers = collect((edges[1:end-1] .+ edges[2:end]) ./ 2)
    dxbin   = step(edges)

    B     = zeros(p.nbins)
    Bgrad = zeros(p.nbins)
    binidx(xi) = clamp(Int(floor((xi - p.xmin) / dxbin)) + 1, 1, p.nbins)
    gauss(u, s) = exp(-u^2 / (2s^2)) / (sqrt(2π) * s)

    fpt     = fill(NaN, p.N)
    escaped = falses(p.N)

    factor  = p.alpha / (1 + p.alpha)
    xgrid, dxg, ygrid, dyg = build_grids(p)
    rho_inf  = marginal_target_density(p.beta, factor, xgrid, dxg, ygrid, dyg)
    l1_times = Float64[]
    l1_vals  = Float64[]
    push!(l1_times, 0.0)
    push!(l1_vals, l1_distance(kde_eval(x, xgrid, p.kde_bandwidth), rho_inf, dxg))

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
            bias = p.alpha * Bgrad[i]
            fx = dUdx(xi, yi)
            fy = dUdy(xi, yi)

            x[k] = reflect(xi + (-fx + bias) * p.dt + sqrt(2 * p.dt / p.beta) * randn(rng), p.xmin, p.xmax)
            y[k] = reflect(yi + (-fy) * p.dt        + sqrt(2 * p.dt / p.beta) * randn(rng), p.ymin, p.ymax)

            if !escaped[k] && x[k] > 0
                escaped[k] = true
                fpt[k] = t
            end
        end

        if step % p.snapshot_every == 0
            push!(l1_times, t)
            push!(l1_vals, l1_distance(kde_eval(x, xgrid, p.kde_bandwidth), rho_inf, dxg))
        end
    end

    return fpt, escaped, l1_times, l1_vals
end

# =======================================================================
# Parallel alpha/gamma sweep, WITH REPLICATES.
#
# For every alpha in ALPHAS, and for the two fixed reference runs
# (Unbiased Langevin, ABF(gamma=1)), we run `n_reps` independent
# replicate simulations (distinct RNG seeds) and average their L1(t)
# curves and escape fractions pointwise. Since dt, nsteps and
# snapshot_every are identical across replicates, all reps of a given
# curve share the exact same l1_times grid, so pointwise averaging of
# l1_vals is well-defined (no interpolation needed).
#
# All (alpha, rep) jobs for ABP+ABF, plus all rep jobs for the two
# reference curves, are flattened into ONE job list and dispatched via
# a single Threads.@threads loop so that work is balanced evenly across
# threads regardless of how n_alpha vs. n_reps trade off.
# =======================================================================
function run_alpha_sweep(; n_alpha::Int = 15, n_reps::Int = 10,
                            N::Int = 300, dt::Float64 = 1e-4, nsteps::Int = 40_000,
                            beta::Float64 = 1.0, sigma_kernel::Float64 = 0.02,
                            kde_bandwidth::Float64 = 0.01, kde_grid_n::Int = 200,
                            ny_quad::Int = 200, ymin::Float64 = -2.5, ymax::Float64 = 2.5,
                            snapshot_every::Int = 100, base_seed::Int = 1000)

    ALPHAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    n = length(ALPHAS)
    println("Alpha sweep over $(n) values: $(ALPHAS), n_reps=$(n_reps) replicate(s) per curve.")
    println("Running with $(Threads.nthreads()) threads.")

    # per-(alpha, rep) storage for ABP / ABF
    l1_times_abp_reps = [Vector{Vector{Float64}}(undef, n_reps) for _ in 1:n]
    l1_vals_abp_reps  = [Vector{Vector{Float64}}(undef, n_reps) for _ in 1:n]
    l1_times_abf_reps = [Vector{Vector{Float64}}(undef, n_reps) for _ in 1:n]
    l1_vals_abf_reps  = [Vector{Vector{Float64}}(undef, n_reps) for _ in 1:n]
    escape_abp_reps   = [Vector{Float64}(undef, n_reps) for _ in 1:n]
    escape_abf_reps   = [Vector{Float64}(undef, n_reps) for _ in 1:n]

    # per-rep storage for the two fixed reference curves
    t_unb_reps  = Vector{Vector{Float64}}(undef, n_reps)
    l1_unb_reps = Vector{Vector{Float64}}(undef, n_reps)
    esc_unb_reps = Vector{Float64}(undef, n_reps)

    t_ref_reps  = Vector{Vector{Float64}}(undef, n_reps)
    l1_ref_reps = Vector{Vector{Float64}}(undef, n_reps)
    esc_ref_reps = Vector{Float64}(undef, n_reps)

    # Job types, flattened into one list for balanced parallel dispatch:
    #   (:sweep, alpha_idx, rep_idx)  -> runs both ABP(alpha) and ABF(gamma) for this rep
    #   (:unbiased, rep_idx)          -> unbiased reference, this rep
    #   (:refgamma1, rep_idx)         -> ABF(gamma=1) reference, this rep
    jobs = Tuple{Symbol,Int,Int}[]
    for i in 1:n, r in 1:n_reps
        push!(jobs, (:sweep, i, r))
    end
    for r in 1:n_reps
        push!(jobs, (:unbiased, 0, r))
        push!(jobs, (:refgamma1, 0, r))
    end
    njobs = length(jobs)
    println("Total parallel jobs: $(njobs)")

    Threads.@threads for jidx in 1:njobs
        kind, i, r = jobs[jidx]

        if kind == :sweep
            a = Float64(ALPHAS[i])
            # Unique, reproducible seed per (alpha, rep); distinct streams for ABP vs ABF.
            seed_abp = base_seed + 2 * ((i - 1) * n_reps + r)
            seed_abf = seed_abp + 1

            p_abp = Params(N = N, dt = dt, nsteps = nsteps, beta = beta, alpha = a,
                            sigma_kernel = sigma_kernel, x0 = -1.0, y0 = 0.0, seed = seed_abp,
                            kde_bandwidth = kde_bandwidth, kde_grid_n = kde_grid_n,
                            ny_quad = ny_quad, ymin = ymin, ymax = ymax,
                            snapshot_every = snapshot_every)
            p_abf = Params(N = N, dt = dt, nsteps = nsteps, beta = beta, alpha = a,
                            sigma_kernel = sigma_kernel, x0 = -1.0, y0 = 0.0, seed = seed_abf,
                            kde_bandwidth = kde_bandwidth, kde_grid_n = kde_grid_n,
                            ny_quad = ny_quad, ymin = ymin, ymax = ymax,
                            snapshot_every = snapshot_every)

            _, esc_abp, t_abp, l1_abp = run_ABP_fpt(p_abp)
            _, esc_abf, t_abf, l1_abf = run_ABF_fpt(p_abf)   # gamma = alpha/(1+alpha)

            l1_times_abp_reps[i][r] = t_abp;  l1_vals_abp_reps[i][r] = l1_abp
            l1_times_abf_reps[i][r] = t_abf;  l1_vals_abf_reps[i][r] = l1_abf
            escape_abp_reps[i][r] = mean(esc_abp)
            escape_abf_reps[i][r] = mean(esc_abf)

            println("  alpha=$(a) rep=$(r)/$(n_reps) done (thread $(Threads.threadid()))  escaped ABP=$(escape_abp_reps[i][r]), ABF=$(escape_abf_reps[i][r])")

        elseif kind == :unbiased
            seed = base_seed + 1_000_000 + r
            p_unb = Params(N = N, dt = dt, nsteps = nsteps, beta = beta,
                            sigma_kernel = sigma_kernel, x0 = -1.0, y0 = 0.0, seed = seed,
                            kde_bandwidth = kde_bandwidth, kde_grid_n = kde_grid_n,
                            ny_quad = ny_quad, ymin = ymin, ymax = ymax,
                            snapshot_every = snapshot_every)
            _, esc_unb, t_unb, l1_unb = run_unbiased_fpt(p_unb)
            t_unb_reps[r] = t_unb; l1_unb_reps[r] = l1_unb; esc_unb_reps[r] = mean(esc_unb)
            println("  Unbiased rep=$(r)/$(n_reps) done (thread $(Threads.threadid()))  escaped=$(esc_unb_reps[r])")

        else # :refgamma1
            seed = base_seed + 2_000_000 + r
            p_ref = Params(N = N, dt = dt, nsteps = nsteps, beta = beta, alpha = 1.0,
                            sigma_kernel = sigma_kernel, x0 = -1.0, y0 = 0.0, seed = seed,
                            kde_bandwidth = kde_bandwidth, kde_grid_n = kde_grid_n,
                            ny_quad = ny_quad, ymin = ymin, ymax = ymax,
                            snapshot_every = snapshot_every)
            _, esc_ref, t_ref, l1_ref = run_ABF_fpt(p_ref; gamma_override = 1.0)
            t_ref_reps[r] = t_ref; l1_ref_reps[r] = l1_ref; esc_ref_reps[r] = mean(esc_ref)
            println("  ABF(gamma=1) ref rep=$(r)/$(n_reps) done (thread $(Threads.threadid()))  escaped=$(esc_ref_reps[r])")
        end
    end

    # ---- average across replicates (pointwise; time grids match exactly) ----
    avg_curve(vecs) = mean(hcat(vecs...), dims = 2)[:, 1]

    l1_times_abp    = Vector{Vector{Float64}}(undef, n)
    l1_vals_abp     = Vector{Vector{Float64}}(undef, n)
    l1_times_abf    = Vector{Vector{Float64}}(undef, n)
    l1_vals_abf     = Vector{Vector{Float64}}(undef, n)
    escape_frac_abp = Vector{Float64}(undef, n)
    escape_frac_abf = Vector{Float64}(undef, n)

    for i in 1:n
        l1_times_abp[i] = l1_times_abp_reps[i][1]
        l1_vals_abp[i]  = avg_curve(l1_vals_abp_reps[i])
        l1_times_abf[i] = l1_times_abf_reps[i][1]
        l1_vals_abf[i]  = avg_curve(l1_vals_abf_reps[i])
        escape_frac_abp[i] = mean(escape_abp_reps[i])
        escape_frac_abf[i] = mean(escape_abf_reps[i])
    end

    t_unb = t_unb_reps[1]
    l1_unb = avg_curve(l1_unb_reps)
    escape_frac_unb = mean(esc_unb_reps)

    t_ref = t_ref_reps[1]
    l1_ref = avg_curve(l1_ref_reps)
    escape_frac_ref = mean(esc_ref_reps)

    return (ALPHAS = ALPHAS, n_reps = n_reps,
            l1_times_abp = l1_times_abp, l1_vals_abp = l1_vals_abp,
            l1_times_abf = l1_times_abf, l1_vals_abf = l1_vals_abf,
            escape_frac_abp = escape_frac_abp, escape_frac_abf = escape_frac_abf,
            t_unb = t_unb, l1_unb = l1_unb, escape_frac_unb = escape_frac_unb,
            t_ref = t_ref, l1_ref = l1_ref, escape_frac_ref = escape_frac_ref)
end

# ---------------------------------------------------------------------
# Figure: 3 panels (Unbiased | ABP(alpha) | ABF(gamma)), L1(t) curves
# colored by alpha, escape fraction in each legend label. Curves are
# now averages over res.n_reps replicates (noted in the plot title).
# ---------------------------------------------------------------------
function build_sweep_l1_figure(res)
    ALPHAS = res.ALPHAS
    n = length(ALPHAS)
    logA = log10.(ALPHAS)
    lo, hi = extrema(logA)
    colors = [cgrad(:viridis)[(logA[i] - lo) / max(hi - lo, eps())] for i in 1:n]

    esc_pct_unb = round(Int, 100 * res.escape_frac_unb)
    p0 = plot(title = "Unbiased Langevin", xlabel = "t", ylabel = "L1 distance", yscale = :log10,
              legend = :outertopright, legendfontsize = 6)
    #plot!(p0, res.t_unb, max.(res.l1_unb, 1e-12), color = :gray, lw = 2,
    #      label = "Unbiased (esc $(esc_pct_unb)%)")

    p1 = plot(title = "ABP(alpha)", xlabel = "t", ylabel = "L1 distance", yscale = :log10,
              legend = :outertopright, legendfontsize = 6)
    p2 = plot(title = "ABF(gamma=alpha/(1+alpha))", xlabel = "t", ylabel = "L1 distance", yscale = :log10,
              legend = :outertopright, legendfontsize = 6)

    for i in 1:n
        esc_pct_abp = round(Int, 100 * res.escape_frac_abp[i])
        esc_pct_abf = round(Int, 100 * res.escape_frac_abf[i])
        plot!(p1, res.l1_times_abp[i], max.(res.l1_vals_abp[i], 1e-12),
              color = colors[i], lw = 1.5, label = "α=$(ALPHAS[i]) (esc $(esc_pct_abp)%)")
        plot!(p2, res.l1_times_abf[i], max.(res.l1_vals_abf[i], 1e-12),
              color = colors[i], lw = 1.5, label = "")
    end

    esc_pct_ref = round(Int, 100 * res.escape_frac_ref)
    plot!(p2, res.t_ref, max.(res.l1_ref, 1e-12), color = :black, lw = 3, ls = :dash,
          label = "γ=1 ref. (esc $(esc_pct_ref)%)")
    plot!(p1, res.t_unb, max.(res.l1_unb, 1e-12), color = :gray, lw = 2,
          label = "Unbiased (esc $(esc_pct_unb)%)")
    plot!(p2, res.t_unb, max.(res.l1_unb, 1e-12), color = :gray, lw = 2,
          label = "Unbiased (esc $(esc_pct_unb)%)")

    plot(p1, p2, layout = (1, 2), size = (1300, 600),
         plot_title = "2D gate potential — L1 convergence over time (κ=$(KAPPA), σ=$(SIGMA), β=1.0, avg of $(res.n_reps) reps)")
end

function main_sweep()

    N_REPS = 10   # <-- number of independent replicate runs averaged per curve (try 10 or 20)

    res = run_alpha_sweep(n_alpha = 10, n_reps = N_REPS, N = 500, dt = 1e-4, nsteps = 50_000,
                            beta = 1.0, sigma_kernel = 0.02, kde_bandwidth = 0.01,
                            kde_grid_n = 200, ny_quad = 200, ymin = -2.5, ymax = 2.5,
                            snapshot_every = 20, base_seed = 800)

    fig = build_sweep_l1_figure(res)
    savefig(fig, "alpha_sweep_l1_vs_time_2d.html")
    println("Saved alpha_sweep_l1_vs_time_2d.html")
end

main_sweep()