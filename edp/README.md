# Discretization of the ABF / ABP Fokker–Planck equations

Finite-element solver for the density-level description of the ABF and ABP
algorithms. The model and the biasing drifts are detailed in the top-level
`README.md`; this note specifies the discretization scheme. Implementation in FreeFEM
(P1), Python/matplotlib for figures.

## 1. Continuous problem

On the box $\Omega=[-x_{\max},x_{\max}]^2$ (coordinates $(x,y)$, with reaction coordinate
$x$), the density solves a conservative Fokker–Planck equation with a biasing force
$g(x)$ acting along the direction $x$:

```math
\partial_t\rho = \nabla\cdot J,\qquad
J = \rho\,\nabla U + \beta^{-1}\nabla\rho + \rho\, g\, e_x ,\qquad
J\cdot {\rm n} = 0 \text{ on } \partial\Omega .
```

```math
U(x,y) = (x^2-1)^2 + k(x)^{d-1} y^2,\quad k(x)=1+\kappa_0 e^{-x^2/2\sigma^2},\qquad
F(x) = (x^2-1)^2 + \frac{d-1}{2\beta}\ln k(x) + \text{const}.
```

```math
\text{ABF:}\quad g = -\gamma\,A'(x),\, A'(x)=\frac{\int\partial_x U\,\rho\,dy}{\int\rho\,dy},\, \gamma=\frac{\alpha}{\alpha+1};
\qquad
\text{ABP:}\quad g = \frac{\alpha}{\beta}\,\partial_x\ln\rho^1,\, \rho^1(x)=\int\rho\,dy .
```

Both flows share the explicit stationary state
$\rho_\infty \propto \exp\left(-\beta(U-\gamma F)\right)$, implying the same
marginal $\rho^1_\infty\propto e^{-\beta F/(\alpha+1)}$.
Defaults: $\beta=1$, $\kappa_0=1$, $\sigma=0.25$, $x_{\max}=2.5$; $\alpha$
and $d$ are the swept parameters.

## 2. Space

- Fixed structured mesh `square(nx,nx)` on $\Omega = [-x_{\max},x_{\max}]^2$, with a space $V_2$ of P1 elements; no re-adaptation (refine with `-nx` parameter if unstabilities appear).
- **No-flux Neumann** at the boundary of the domain gives exact mass conservation up to numerical precision. To avoid numerical drift, the density is clipped to positive values and renormalized at each time step.
- **Matrix representations of marginals.** We fix a space $V_1$ of P1 elements on a mesh over $\Omega^1=[-x_{\max},x_{\max}]$ and
  the extension `R = interpolate(Vh2,Vh1)` (sends a 1D function to a 2D function constant in $y$; exact
  on the structured mesh). Marginalization has the weak representation

  ```math
  \int_{\Omega^1} \rho^1(x) \varphi(x)\,{\rm d} x = \int_{\Omega}\rho(x,y)(R\varphi)(x,y)\,{\rm d}x{\rm d}y,\qquad \forall \varphi
  ```

  or, with mass matrix $M_i$ for the $V_i$ $L^2$-inner product, this can be written matricially

  ```math
  \rho^1 = M_1^{-1} R^{\mathsf T} M_2\, \rho.
  ```

  We also define the operator

  ```math
    P = R\,M_1^{-1}R^{\mathsf T}M_2,
  ```
  
  which sends $\rho$ to $\rho^1$ and re-extends to a $2D$ function.

## 3. Time

We use an implicit scheme in which the biasing nonlinearity is explicit/lagged:
every nonlinear quantity is evaluated at the known $\rho^n$. The unknown $\rho^{n+1}\in V_2$
enters linearly, so each time step requires a linear solve: find $\rho^{n+1}$
such that, for all $v\in V_2$,

```math
\frac{1}{\Delta t}\int(\rho^{n+1}-\rho^n)\,v
+\int\big(\rho^{n+1}\nabla U+\beta^{-1}\nabla\rho^{n+1}\big)\cdot\nabla v
+b\big(\rho^n;\,\rho^{n+1},\,v\big)=0 .
```

The biasing form $b(\rho^n;\cdot,\cdot)$ is bilinear in its last two arguments (its
coefficients depend only on the explicit $\rho^n$);

Write $[a]$ for the matrix of a bilinear form $a$ on $V_2$, i.e. $[a]_{ij}=a(\varphi_j,\varphi_i)$, so

```math
M_2=\big[\textstyle\int uv\big],\qquad
K_{\nabla U}=\left[\int u\,\nabla U\cdot\nabla v\right],\qquad
K_{\Delta}=\left[\int\nabla u\cdot\nabla v\right].
```

The step is then the linear system

```math
A(\rho^n)\,\rho^{n+1}=\frac{1}{\Delta t}M_2\,\rho^{n},\qquad
A(\rho^n)=\underbrace{\frac{1}{\Delta t}M_2+K_{\nabla U}+\beta^{-1}K_{\Delta}}_{A_0\ \text{(constant)}}+B(\rho^n),
```

where $K_{\nabla U}+\beta^{-1}K_{\Delta}$ is the Galerkin matrix of the unbiased overdamped
Fokker–Planck operator. $A_0$ is constant in time — assembled and factored **once** — and
only $B(\rho^n)$ is rebuilt each step. ABF and ABP **differ only in $B(\rho^n)$**.
Both solve the system by GMRES preconditioned by $A_0$, applying $B(\rho^n)$ as a matrix-free linear operator.

**ABF.** The bias advects with the lagged mean force $g^n=-\gamma A'^{\,n}$, where
$A'^{\,n}=(\partial_x U\,\rho^n)^1/\max\left((\rho^n)^1,\varepsilon\right)$ ($\varepsilon=10^{-10}$
prevents error from underflows where the flux $\rho g$ vanishes). For all $v$,

```math
b_{\mathrm{ABF}}(\rho^n;\rho^{n+1},v)=\int\rho^{n+1}\,g^n\,\partial_x v,
\qquad
B(\rho^n)=\left[\int u\,g^n\,\partial_x v\right].
```

**ABP.** The velocity $\tfrac{\alpha}{\beta}\partial_x\ln\rho^1$ is unbounded, but the flux
$\frac{\alpha}{\beta}\rho\,\partial_x\ln\rho^1=\frac{\alpha}{\beta}\,c\,\partial_x\rho^1$
carries the better-behaved conditional density $c=\rho/\rho^1$. Lagging $c^n=\rho^n/(\rho^n)^1$ but
keeping the gradient of the marginal implicit through the extend-marginal map
$P=R M_1^{-1}R^{\mathsf T}M_2$ keeps the form bilinear: for all $v$,

```math
b_{\mathrm{ABP}}(\rho^n;\rho^{n+1},v)=\frac{\alpha}{\beta}\int c^n\,\partial_x\left(P\rho^{n+1}\right)\,\partial_x v,
\qquad
B(\rho^n)=\frac{\alpha}{\beta}\,S({c^n})P,\quad
S({c^n}):=\left[\int c^n\,\partial_x u\,\partial_x v\right].
```

Since $P$ couples the unknown's marginal into every node, $B(\rho^n)$ is dense and is never
assembled: the GMRES operator applies $S({c^n})P$ as sparse mat-vecs plus one $M_1^{-1}$ solve from $P$.

## 4. Initial condition and output
- A Gaussian in the left well, $\rho_0\propto e^{-((x+1)^2+y^2)/2s_0^2}$,
normalized to unit mass; identical for both methods.
- Mass $\int\rho\equiv1$ (to $\sim10^{-14}$) and $\min\rho\ge0$ by construction.
- Each driver records $\|\rho_t-\rho_\infty\|_{L^1}$ per step (`*_trace.txt`)
- Per frame the density $\rho$, the marginal $\rho^1$, and the bias $g(x)$ are dumped.

## 5. Files and usage

Data → `sim_results/`, figures → `plots/`, both tagged by $(\alpha,d)$

| file | role |
|---|---|
| `entropic_potential.idp` | the **model**: $\beta,\kappa_0,\sigma,d,x_{\max}$ and `func U, dxU, dyU, Fx`. Swap this file for another potential. |
| `solver.idp` | potential-agnostic numerics: mesh, P1 spaces, marginal operators ($R,M_1,M_2$, `marginal`), `clampNormalize`, the shared $A_0$/`Mdt`/`Prec`, $\rho_\infty$, $\rho_0$, snapshot storage, run parameters, banner. |
| `abf.edp`, `abp.edp` | drivers: include both `.idp`, define only the step operator ($B_g$ resp. $S^{c}P$) and the time loop. |
| `write_frames.idp` | dumps mesh + per-frame $(\rho,\rho^1,g)$ + params + trace. |
| `make_plots.py` | all figures for every run in `sim_results/`: 3-panel movie (density + marginal + bias), $L^1$ trace, stacked comparison, $g(x,t)$ space-time. `--fps N`, `--alpha A --d D`. |
| `run_sweep.sh` | bounded-parallel sweep over $d$ and $\alpha$. |

FreeFEM CLI: `-d -nx -dt -T -stride -beta -kappa0 -sigma -alphaTemp -s0`.

```
for D in 2 5 10; do
  FreeFem++ -nw abf.edp -nx 250 -T 5 -dt 0.001 -stride 20 -d $D -alphaTemp 5
  FreeFem++ -nw abp.edp -nx 250 -T 5 -dt 0.001 -stride 20 -d $D -alphaTemp 5
done
python3 make_plots.py --fps 20 --workers 8 --dpi 85        # sim_results/ -> plots/
```
