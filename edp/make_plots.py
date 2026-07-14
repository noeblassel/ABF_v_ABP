#!/usr/bin/env python3
"""Render every figure for every run found in sim_results/.

For each run (tagged by alpha, d) it writes, into plots/:
  <abf|abp>_alpha<A>_d<D>.gif   three-panel movie: density | marginal | biasing force g
  trace_alpha<A>_d<D>.png       L1 distance to the stationary state (ABF vs ABP)
  comparison_alpha<A>_d<D>.gif  ABF (top) / ABP (bottom) movies stacked
  bias_alpha<A>_d<D>.png        space-time heatmap of the biasing force g(x,t)

Usage:  python3 make_plots.py [--fps N] [--alpha A --d D]
                              [--simdir sim_results] [--plotdir plots]
        With no --alpha/--d it processes every run in sim_results/.
"""
import argparse
import glob
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image, ImageSequence


# ----------------------------- model + IO ---------------------------------
def load_params(path):
    p = {}
    with open(path) as f:
        for line in f:
            k, v = line.split()
            p[k] = v
    return p


def model(p):
    """Analytic U, F, F' and the tempered target, from a params dict."""
    beta = float(p["beta"]); k0 = float(p["kappa0"]); s = float(p["sigma"])
    d = float(p["d"]); al = float(p["alphaTemp"])
    aeff = lambda x: (1.0 + k0 * np.exp(-x**2 / (2 * s**2)))**(d - 1.0)
    U = lambda x, y: (x**2 - 1.0)**2 + aeff(x) * y**2
    F = lambda x: (x**2 - 1.0)**2 - np.log(4 * np.pi / (beta * aeff(x))) / (2 * beta)

    def dF(x):                                   # F'(x) -> target biasing force -gamma F'
        kp = 1.0 + k0 * np.exp(-x**2 / (2 * s**2))
        dkp = -k0 * x / s**2 * np.exp(-x**2 / (2 * s**2))
        return 4 * x * (x**2 - 1.0) + (d - 1.0) / (2 * beta) * dkp / kp

    return dict(beta=beta, d=d, al=al, gamma=al / (al + 1.0),
                xmax=float(p["xmax"]), aeff=aeff, U=U, F=F, dF=dF)


def load_run(base):
    p = load_params(base + "_params.txt")
    c = np.loadtxt(base + "_coords.txt")
    tris = np.loadtxt(base + "_tris.txt", dtype=int)
    xg = np.loadtxt(base + "_xgrid.txt")
    times = np.atleast_1d(np.loadtxt(base + "_times.txt"))
    rho = np.loadtxt(base + "_rho.txt")
    marg = np.loadtxt(base + "_marg.txt")
    bias = np.loadtxt(base + "_bias.txt")
    if rho.ndim == 1:
        rho, marg, bias = rho[None], marg[None], bias[None]
    return dict(p=p, m=model(p), xv=c[:, 0], yv=c[:, 1],
                tri=mtri.Triangulation(c[:, 0], c[:, 1], tris),
                xg=xg, order=np.argsort(xg), times=times,
                rho=rho, marg=marg, bias=bias, base=base)


# ------------------------------- renderers --------------------------------
ULEVELS = [0.3, 0.8, 1.5, 3.0, 5.0, 8.0]


def _targets(run):
    m = run["m"]; xs = run["xg"][run["order"]]
    g = np.exp(-m["beta"] * m["F"](xs) / (m["al"] + 1.0))
    tgt = g / np.trapezoid(g, xs)                 # tempered marginal
    biastgt = -m["gamma"] * m["dF"](xs)           # converged biasing force -gamma F'
    return xs, tgt, biastgt


def render_movie(run, method, outname, plotdir, fps):
    m = run["m"]; tri = run["tri"]; o = run["order"]; t = run["times"]
    rho, marg, bias = run["rho"], run["marg"], run["bias"]
    xmax = m["xmax"]
    xs, tgt, biastgt = _targets(run)
    Uv = m["U"](run["xv"], run["yv"])
    vmax = np.quantile(rho[rho > 0], 0.999) if np.any(rho > 0) else rho.max()
    mmax = max(marg.max(), tgt.max()) * 1.15
    bmax = max(np.abs(bias).max(), np.abs(biastgt).max()) * 1.1

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15, 4.6))
    tcf = a0.tricontourf(tri, np.clip(rho[0], 0, vmax), levels=40, cmap="magma",
                         vmin=0, vmax=vmax, zorder=0)
    a0.tricontour(tri, Uv, levels=ULEVELS, colors="white", alpha=0.35,
                  linewidths=0.7, zorder=1)
    a0.set_aspect("equal"); a0.set_xlabel("x"); a0.set_ylabel("y")
    a0.set_title(r"density $\rho(t,x,y)$")
    fig.colorbar(tcf, ax=a0, shrink=0.85)

    (lm,) = a1.plot(xs, marg[0][o], lw=2, color="#1f77b4", label=r"$\rho^1(t,x)$")
    a1.plot(xs, tgt, "--", color="#d62728", lw=1.6, label=r"target $e^{-\beta F/(\alpha+1)}$")
    a1.set_xlim(-xmax, xmax); a1.set_ylim(0, mmax); a1.set_xlabel("x")
    a1.set_title(r"marginal $\rho^1$"); a1.legend(loc="upper right", fontsize=8)

    (lb,) = a2.plot(xs, bias[0][o], lw=2, color="#2ca02c", label=r"$g(t,x)$")
    a2.plot(xs, biastgt, "--", color="#d62728", lw=1.6, label=r"target $-\gamma F'(x)$")
    a2.set_xlim(-xmax, xmax); a2.set_ylim(-bmax, bmax); a2.set_xlabel("x")
    a2.set_title("biasing force $g$"); a2.legend(loc="upper right", fontsize=8)

    sup = fig.suptitle("")

    def upd(i):
        nonlocal tcf
        tcf.remove()
        tcf = a0.tricontourf(tri, np.clip(rho[i], 0, vmax), levels=40, cmap="magma",
                             vmin=0, vmax=vmax, zorder=0)
        lm.set_ydata(marg[i][o]); lb.set_ydata(bias[i][o])
        sup.set_text(f"{method.upper()}   t = {t[i]:.2f}"
                     f"   (β={m['beta']:g}, d={m['d']:g}, α={m['al']:g})")
        return lm, lb

    out = f"{plotdir}/{outname}.gif"
    FuncAnimation(fig, upd, frames=len(t), interval=80, blit=False).save(
        out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


def render_trace(abf, abp, out):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, run, c in [("ABF", abf, "#1f77b4"), ("ABP", abp, "#d62728")]:
        tr = np.loadtxt(run["base"] + "_trace.txt")
        a0.plot(tr[:, 0], tr[:, 1], color=c, lw=2, label=name)
        a1.semilogy(tr[:, 0], tr[:, 1], color=c, lw=2, label=name)
    for ax in (a0, a1):
        ax.set_xlabel("t"); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_ylabel(r"$\|\rho_t - \rho_\infty\|_{L^1}$")
    a0.set_title("L1 distance to stationary state"); a1.set_title("same, semilog")
    m = abf["m"]
    fig.suptitle(r"$\rho_\infty \propto \exp(-\beta(U - \frac{\alpha}{\alpha+1}F))$"
                 + f"   ($\\alpha$={m['al']:g}, $d$={m['d']:g})", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def render_bias_spacetime(abf, abp, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    # robust symmetric scale: the ABP log-derivative spikes hard at t=0, so clip the
    # top 2% of |g| to keep the bulk evolution legible (extremes saturate).
    allb = np.concatenate([np.abs(abf["bias"]).ravel(), np.abs(abp["bias"]).ravel()])
    vmax = np.percentile(allb, 98) or allb.max()
    pcm = None
    for ax, name, run in [(axs[0], "ABF", abf), (axs[1], "ABP", abp)]:
        o = run["order"]; xs = run["xg"][o]
        pcm = ax.pcolormesh(xs, run["times"], run["bias"][:, o], shading="auto",
                            cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{name}:  $g(x,t)$"); ax.set_xlabel("x")
    axs[0].set_ylabel("t")
    fig.colorbar(pcm, ax=axs, shrink=0.9, label="biasing force $g$")
    m = abf["m"]
    fig.suptitle(f"biasing force evolution   ($\\alpha$={m['al']:g}, $d$={m['d']:g})",
                 fontsize=11)
    fig.savefig(out, dpi=110); plt.close(fig)


def render_comparison(abf_gif, abp_gif, out, fps):
    def frames(path):
        im = Image.open(path)
        return [f.convert("RGBA") for f in ImageSequence.Iterator(im)]
    fa, fb = frames(abf_gif), frames(abp_gif)
    n = min(len(fa), len(fb))
    stacked = []
    for i in range(n):
        A, B = fa[i], fb[i]                        # ABF on top, ABP below
        canvas = Image.new("RGBA", (max(A.width, B.width), A.height + B.height),
                           (255, 255, 255, 255))
        canvas.paste(A, (0, 0)); canvas.paste(B, (0, A.height))
        stacked.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=256))
    stacked[0].save(out, save_all=True, append_images=stacked[1:],
                    duration=int(round(1000 / fps)), loop=0, disposal=2)


# --------------------------------- main -----------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=float, default=14, help="frames/sec for the GIFs (default 14)")
    ap.add_argument("--simdir", default="sim_results")
    ap.add_argument("--plotdir", default="plots")
    ap.add_argument("--alpha", type=float, default=None, help="restrict to this alpha")
    ap.add_argument("--d", type=float, default=None, help="restrict to this d")
    args = ap.parse_args()
    os.makedirs(args.plotdir, exist_ok=True)

    # discover runs: (alpha, d) -> {method: base}
    runs = {}
    for pf in glob.glob(f"{args.simdir}/*_alpha*_d*_params.txt"):
        name = os.path.basename(pf)[:-len("_params.txt")]
        mobj = re.match(r"(abf|abp)_alpha([0-9.]+)_d([0-9.]+)$", name)
        if not mobj:
            continue
        method, a, dd = mobj.groups()
        if (args.alpha is not None and float(a) != args.alpha) or \
           (args.d is not None and float(dd) != args.d):
            continue
        runs.setdefault((a, dd), {})[method] = f"{args.simdir}/{name}"
    if not runs:
        sys.exit(f"no runs found in {args.simdir}/")

    for (a, dd), methods in sorted(runs.items()):
        runtag = f"alpha{a}_d{dd}"
        loaded = {}
        for method, base in sorted(methods.items()):
            run = load_run(base)
            loaded[method] = run
            print("wrote", render_movie(run, method, f"{method}_{runtag}", args.plotdir, args.fps))
        if "abf" in loaded and "abp" in loaded:
            render_trace(loaded["abf"], loaded["abp"], f"{args.plotdir}/trace_{runtag}.png")
            print(f"wrote {args.plotdir}/trace_{runtag}.png")
            render_bias_spacetime(loaded["abf"], loaded["abp"], f"{args.plotdir}/bias_{runtag}.png")
            print(f"wrote {args.plotdir}/bias_{runtag}.png")
            render_comparison(f"{args.plotdir}/abf_{runtag}.gif",
                              f"{args.plotdir}/abp_{runtag}.gif",
                              f"{args.plotdir}/comparison_{runtag}.gif", args.fps)
            print(f"wrote {args.plotdir}/comparison_{runtag}.gif")


if __name__ == "__main__":
    main()
