# Mathematical Background

Local Fourier Analysis was first used by Brandt [1] to analyze the convergence of multi-level adaptive techniques for solving PDEs discretized with finite differences, but the technique has been adapted for multi-level and multi-grid techniques more broadly with finite element discretizations.

By way of example, we will explore Local Fourier Analysis for iterative solvers and mutli-grid methods for the two dimensional Poisson problem, given by the PDE

```math
- \nabla^2 u = f.
```

## Local Fourier Modes

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

Selecting a finite element basis, we can discretize the operator and produce a Toeplitz operator ``A`` . Assuming a H1 Lagrange basis of polynomial order ``p``, the assembled element matrix is of the form

```math
A_e = [a_{i, j}]
```

where ``i, j \in \lbrace 0, 1, 2, \dots, p \rbrace``.
This operator can be diagonalized by the Fourier functions ``\varphi \left(\mathbf{\theta}, \mathbf{x} \right) = e^{\imath \mathbf{\theta} \cdot \mathbf{x} / h}``.

Thus, we have the symbol matrix

```math
\tilde{A} \left( \mathbf{\theta} \right) = 
```

The Fourier modes can thus be represented as

```math
S \left( \mathbf{\theta}, \mathbf{x} ) = [a_{i, j} e^{\imath \mathbf{\theta} \cdot \mathbf{x} / h}].
```

However, notice that ``a_{i, j} e^{\imath \mathbf{\theta} \cdot \mathbf{x} / h}`` and ``a_{k, l} e^{\imath \mathbf{\theta} \cdot \mathbf{x} / h}`` are equivalent when ``i = k \left( mod p \right)`` and ``j = l \left( mod p \right)``.

Do I want to include an explanation as to why this is the Fourier basis?
I will need to put that somewhere eventually.

Really this sentence is a note to users. I'm not sure where to put it, but it should be included.
Perhaps in a yet to be formed code example section?

This Local Fourier Analysis toolkit uses the algebraic representation of PDE operators discussed in [2].

## Jacobi Smoother

M is not based on theta here, but S = I - M^-1 A is

## Multigrid

Here, P and R are based on theta as well. Will want to describe with any smoother dropped into the multigrid (as in the code).

## Custom Smoothers

Example of user defined smoother. User provides element matrix and code maps to symbols.
