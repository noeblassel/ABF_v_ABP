# ABF versus ABP

This repository implements various comparisons between the adaptive biasing force (**ABF**[^LROS]) and the adaptive biasing potential (**ABP**[^MLL]) enhanced sampling algorithms, specifically Markovian versions of these methods where an instantaneous bias is learnt by a population of interacting replicas of the physical system.

## Physical system

### Reduction of a $d$-dimensional model

The model is a caricature of a system displaying **entropic barriers**, which are common in materials and biological systes. We start from a potential on $\mathbb{R}^d$,

$$
U_d(x_1, x_\perp) = W(x_1) + \tfrac{1}{2}\, k(x_1)^2\, |x_\perp|^2,
\qquad x_\perp = (x_2,\dots,x_d) \in \mathbb{R}^{d-1},
$$

with $W(x_1) = (x_1^2-1)^2$ a double well along the reaction coordinate $x_1$, and $k(x_1) = \bigl(1+\kappa_0{\rm e}^{-x_1^2/2\sigma^2}\bigr)^{1/2}$ a transverse *stiffness profile* that has a bump at $x_1 = 0$. Where the confinement is tighter, fewer transverse configurations are accessible: this creates an *entropic bottleneck* near $x_1=0$.

Passing to cylindrical coordinates $x_\perp \mapsto (r,\theta)$ with $r = |x_\perp| \in \mathbb{R}_+$ and $\theta \in \mathbb{S}^{d-2}$, the angular variable can be integrated out, leaving an effective description in $(x_1, r)$. The associated free energy along $x_1$ acquires an entropic contribution
$$
F(x_1) = W(x_1) + \frac{d-1}{\beta}\,\log k(x_1) + \text{const},
$$
so the dimension $d$ controls the height of the entropic barrier. We keep $d$ as an *effective dimension* parameter, folding the effect of the $(d-1)$-transverse degrees of freedom in the stiffness of a single transverse coordinate $r$. This gives the two-dimensional potential

$$
U(x,r) = (x^2-1)^2 + \bigl(1+\kappa_0{\rm e}^{-\frac{x^2}{2\sigma^2}}\bigr)^{d-1} r^2 ,
$$

whose free energy along $x$ reproduces the $d$-dimensional one. The parameter $(d-1)/\beta$ controls the energy/entropy balance in the free energy.

### Reference dynamics

The overdamped Langevin dynamics
$$
{\rm d} (X,R)_t = -\nabla U(X_t,R_t)\,{\rm d} t + \sqrt{2\beta^{-1}}\,{\rm d} W_t
$$

samples the Gibbs measure $\propto {\rm e}^{-\beta U}$. Moreover, it reproduces exactly the $(x,r)$-dynamics of the high-dimensional overdamped Langevin dynamics associated with $U_d$ (the path laws can be shown to be equal).

However, it is *metastable*: the entropic bottleneck obstructs transitions along the reaction coordinate $x$. Enhanced sampling accelerates these transitions by adding a biasing drift in the reaction-coordinate direction $e_1 = e_x$.

## Adaptive biasing methods

Both methods modify the reference dynamics by an extra drift along $e_1$; they differ only in how that drift is built from the current state of the system. Writing $x = X_t \cdot e_1$ for the instantaneous reaction coordinate, and choosing any $\alpha>0$, the **idealized** drift terms are defined as follows.s

- In **ABF**, we add (a fraction of) the *local mean force*,
$$
b^{\mathrm{ABF}}_t = \frac{\alpha}{\alpha+1}\mathbb{E}\bigl[\partial_x U(X_t,R_t) \,\big|\, X_t\cdot e_1 = x\bigr]\, e_1 ,
$$
the conditional average of the local force in the reaction-coordinate direction. At stationarity it equals $F'(x)$, and the corresponding drift cancels the mean force, flattening $F$.

- In **ABP**, we add a multiple of the gradient of a *log-marginal*,
$$
b^{\mathrm{ABP}}_t = -\frac{\alpha}{\beta}\, \partial_x \log \rho^1_t(x)\, e_1 ,
$$
where $\rho^1_t$ is the current law of the reaction coordinate.

In both cases the bias depends on the *law* of the process at time $t$ — a conditional expectation for ABF, a marginal for ABP — which makes the dynamics nonlinear.

In both cases, the stationary measure is a Gibbs measure $\propto \exp(-\beta(U-\gamma F))$, where $\gamma = \alpha/(\alpha+1)$. Increasing $\alpha$ therefore flattens the free-energy barrier.

## Three models

The behavior of these algorithms are are studied at three levels of comparison.s

1. **Nonlinear PDE** (`EDP/`). This corresponds to the evolution of the time-marginals in each method, solved using non-linear PDE solvers, implemented in FreeFEM++

2. **McKean–Vlasov SDE** (`mean_field/`). This correspond to the mean-field limit of the algorithm, the stochastic dynamics of a typical replica in the infinite population limit of the algorithm.

3. **Interacting particle system** (`IPS/`). This is the actual enhanced sampling scheme, given by an interacting particle system. The biasing term is estimated from the instantaneous state of the population.

## List of subdirectories

- `EDP/` — nonlinear PDE models
- `Mean-field/` — McKean–Vlasov diffusion model
- `IPS/` — interacting particle system

## References

[^LROS]: T. Lelièvre, F. Otto, M. Rousset, and G. Stoltz, *Long-time convergence of an Adaptive Biasing Force method*, Nonlinearity **21** (2008), no. 6, 1155–1181. [doi:10.1088/0951-7715/21/6/001](https://doi.org/10.1088/0951-7715/21/6/001), [arXiv:0706.1695](https://arxiv.org/abs/0706.1695).

[^MLL]: T. Lelièvre, X. Lin, and P. Monmarché, *Convergence rates for an Adaptive Biasing Potential scheme from a Wasserstein optimization perspective*, preprint (2025). [arXiv:2501.17979](https://arxiv.org/abs/2501.17979).
