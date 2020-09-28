# Mathematical Background

Local Fourier Analysis was first used by Brandt [1] to analyze the convergence of multi-level adaptive techniques for solving PDEs discretized with finite differences, but the technique has been adapted for multi-level and multi-grid techniques more broadly with finite element discretizations.

By way of example, we will explore Local Fourier Analysis for iterative solvers and mutli-grid methods for the two dimensional Poisson problem.

Consider the PDE

```math
- \nabla^2 u = f.
```

In traditional Fourier Analysis, we consider the eigenfunctions and corresponding eigenvalues of the diffusion operator, ``\nabla^2 u``, ``\varphi^{l, m} \left( \mathbf{x} \right) = e^{\left( l \pi x_1 \right)} e^{\left( m \pi x_2 \right)}`` and ``\lambda^{l, m} = l^2 + m^2``, for ``l, m \in \mathcal{Z}``.
In contrast, Local Fourier Analysis considers the local properties of the descretized system via its eigenfunction and eigenvalues.

The Poisson problem has the weak formulation

```math
\int_{\Omega} \nabla u \cdot \nabla v - \int_{\partial \Omega} \nabla u v = \int_{\Omega} f v, \forall v \in V
```

for some suitable ``V \subseteq H_0^1 \left( \Omega \right)``.

The bilinear formulation for the diffusion operator associated with this weak form is given by

```math
a \left( u, v \right) = \int_{\Omega} \nabla u \cdot \nabla v.
```

This bilinear form can be descritized with an arbitrary order finite element basis via

```math
A = B^T D B
```
