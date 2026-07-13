# ABF versus ABP

This repository implements various comparisons between the adaptive biasing force (**ABF**[^LROS]) and the adaptive biasing potential (**ABP**[^MLL]) enhanced sampling techniques, specifically Markovian versions of these algorithms where an instantaneous bias is learnt by a population of interacting replicas of the physical system.

## The toy model

### Origin: reduction of a $d$-dimensional confinement

The model is a low-dimensional caricature of a system with an **entropic barrier**. We start from a potential on $\mathbb{R}^d$,

$$
U_d(x_1, x_\perp) = W(x_1) + \tfrac{1}{2}\, k(x_1)^2\, |x_\perp|^2,
\qquad x_\perp = (x_2,\dots,x_d) \in \mathbb{R}^{d-1},
$$

with $W(x_1) = (x_1^2-1)^2$ a double well along the reaction coordinate $x_1$, and $k(x_1) = \bigl(1+\kappa_0{\rm e}^{-x_1^2/2\sigma^2}\bigr)^{1/2}$ a transverse **stiffness profile** that spikes near $x_1 = 0$. Where the confinement is tight, few transverse configurations are accessible: the bottleneck is *entropic* rather than energetic.

Passing to cylindrical coordinates $x_\perp \mapsto (r,\theta)$ with $r = |x_\perp| \in \mathbb{R}_+$ and $\theta \in \mathbb{S}^{d-2}$, the angular variable decouples and integrates out, leaving an effective description in $(x_1, r)$. The associated free energy along $x_1$ acquires an entropic contribution that is **linear in the number of transverse directions**,

$$
F(x_1) = W(x_1) + \frac{d-1}{\beta}\,\log k(x_1) + \text{const},
$$

so the dimension $d$ controls the height of the entropic barrier. We keep $d$ as a continuous parameter by folding the $(d-1)$-fold effect into the stiffness of a single transverse coordinate $r$, giving the two-dimensional working potential

$$
U(x,r) = (x^2-1)^2 + \bigl(1+\kappa_0{\rm e}^{-\frac{x^2}{2\sigma^2}}\bigr)^{d-1} r^2 ,
$$

whose free energy along $x$ reproduces the $d$-dimensional one. Here $d=2$ recovers the original single-channel model, and larger $d$ raises the entropic barrier.

### Reference dynamics

The overdamped Langevin dynamics

$$
{\rm d} (X,R)_t = -\nabla U(X_t,R_t)\,{\rm d} t + \sqrt{2\beta^{-1}}\,{\rm d} W_t
$$

samples the Gibbs measure $\propto {\rm e}^{-\beta U}$. It is **metastable**: the entropic bottleneck makes transitions along the reaction coordinate $x$ rare. Enhanced sampling accelerates these transitions by adding a biasing drift **in the reaction-coordinate direction $e_1 = e_x$**, so as to flatten the free energy $F$.

## Biasing the reaction coordinate

Both methods modify the reference dynamics by an extra drift along $e_1$; they differ only in how that drift is built from the *current state* of the system. Writing $x = X_t \cdot e_1$ for the instantaneous reaction coordinate,

- **ABF** adds the instantaneous **mean force**,
$$
b^{\mathrm{ABF}}_t = \mathbb{E}\bigl[\partial_x U(X_t,R_t) \,\big|\, X_t\cdot e_1 = x\bigr]\, e_1 ,
$$
the conditional average of the local force in the reaction-coordinate direction. At stationarity it equals $F'(x)$, and the corresponding drift cancels the mean force, flattening $F$.

- **ABP** adds the gradient of a **tempered log-marginal**,
$$
b^{\mathrm{ABP}}_t = -\frac{\alpha}{\beta}\, \partial_x \log \rho^1_t(x)\, e_1 ,
$$
where $\rho^1_t$ is the current law of the reaction coordinate. This penalises already-visited values of $x$ and drives $\rho^1_t$ towards a flatter (higher-temperature) profile.

In both cases the bias depends on the *law* of the process — a conditional expectation for ABF, a marginal for ABP — which makes the dynamics nonlinear.

## Three levels of description

The same mean-field flow is studied at three levels of increasing concreteness, one per subdirectory.

1. **Nonlinear PDE** (`EDP/`). The density $\rho_t(x,r)$ evolves by a nonlinear Fokker–Planck equation: the reference Fokker–Planck operator plus the divergence of the biasing flux, in which the bias is a functional of $\rho_t$ itself. This is the deterministic, density-level picture — the mean-field limit in its most directly computable form.

2. **McKean–Vlasov diffusion** (`Mean-field/`). The same flow written as a single stochastic process whose drift depends on its own law: a self-interacting diffusion whose Fokker–Planck equation is exactly the PDE above. It is the idealised infinite-population limit of the algorithm.

3. **Interacting particle system** (`IPS/`). The actual molecular-dynamics algorithm: $N$ replicas of the system evolve simultaneously, and the mean-field quantity — the conditional mean force for ABF, the reaction-coordinate marginal for ABP — is replaced by an empirical estimate over the replicas. As $N \to \infty$ the particle system converges to the McKean–Vlasov diffusion (propagation of chaos); for finite $N$ it is a practical, implementable enhanced-sampling scheme.

## List of subdirectories

- `EDP/` — nonlinear PDE models
- `Mean-field/` — McKean–Vlasov diffusion model
- `IPS/` — interacting particle system
- `Notes/` — theoretical and technical notes

## References

[^LROS]: T. Lelièvre, F. Otto, M. Rousset, and G. Stoltz, *Long-time convergence of an Adaptive Biasing Force method*, Nonlinearity **21** (2008), no. 6, 1155–1181. [doi:10.1088/0951-7715/21/6/001](https://doi.org/10.1088/0951-7715/21/6/001), [arXiv:0706.1695](https://arxiv.org/abs/0706.1695).

[^MLL]: T. Lelièvre, X. Lin, and P. Monmarché, *Convergence rates for an Adaptive Biasing Potential scheme from a Wasserstein optimization perspective*, preprint (2025). [arXiv:2501.17979](https://arxiv.org/abs/2501.17979).
