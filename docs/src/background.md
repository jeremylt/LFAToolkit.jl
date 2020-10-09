# Mathematical Background

Local Fourier Analysis was first used by Brandt [1] to analyze the convergence of multi-level adaptive techniques for solving PDEs discretized with finite differences, but the technique has been adapted for multi-level and multi-grid techniques using finite element discretizations.

By way of example, we will explore Local Fourier Analysis for multi-level and mutli-grid methods for the two dimensional Poisson problem, given by the PDE

```math
- \nabla^2 u = f.
```

## Local Fourier Modes

Local Fourier Analysis considers the local properties of the descretized system via its Fourier modes and the eigenvalues of the associated symbol matrix.
We will describe the simple one dimensional scalar case and extend it to an arbitrary dimension, degree, and component finite element problem.

First consider a scalar Toeplitz operator ``L_h`` on a one dimensional infinite uniform grid ``G_h``.
This operator is given by

```math
L_h \mathrel{\hat{=}} \left[ s_\kappa \right]_h \left( \kappa \in V \right)

L_h w_h \left( x \right) = \sum_{\kappa \in V} s_\kappa w_h \left( x + \kappa h \right)
```

where ``V \subset \mathcal{Z}`` is a finite index set, ``s_\kappa \in \mathcal{R}`` are constant coefficients, and ``w_h \left( x \right)`` is a ``l^2`` function on ``G_h``.

As ``L_h`` is Toeplitz, it can be diagonalized by the standard Fourier modes ``\varphi \left( \theta, x \right) = e^{\imath \theta x / h}``.

If for all grid functions ``\varphi \left( \theta, x \right)`` we have

```math
S_h \varphi \left( \theta, x \right) = \tilde{L}_h \left( \theta \right) \varphi \left( \theta, x \right)
```

then ``\tilde{L}_h \left( \theta \right) = \sum_{\kappa \in V} s_\kappa e^{\imath \theta \kappa}`` is the **symbol** of ``L_h``.

We can extend this to a ``p \times p`` linear system of operators representing a scalar problem on a ``p`` order finite element

```math
L_h = \begin{bmatrix}
    L_h^{1, 1}  &  \cdots  &  L_h^{1, p}  \\
    \vdots      &  \vdots  &  \vdots      \\
    L_h^{p, 1}  &  \cdots  &  L_h^{p, p}
\end{bmatrix}
```

where ``L_h^{i, j}`` is given by a scalar Toeplitz operator describing how component ``j`` appears in the equation for component ``i``.

Consider the specific case of the Topeliz operator representing the scalar diffusion opperator.
The Poisson problem has the weak formulation

```math
\int_{\Omega} \nabla u \cdot \nabla v - \int_{\partial \Omega} \nabla u v = \int_{\Omega} f v, \forall v \in V
```

for some suitable ``V \subseteq H_0^1 \left( \Omega \right)``.

The bilinear formulation for the diffusion operator associated with this weak form is given by

```math
a \left( u, v \right) = \int_{\Omega} \nabla u \cdot \nabla v.
```

Selecting a finite element basis, we can discretize the operator and produce a Toeplitz operator ``A`` .
Assuming a H1 Lagrange basis of polynomial order ``p`` and using the algebraic representation of PDE operators discussed in [2], the assembled element matrix is of the form

```math
A_e = B^T J^T D J B
```

where ``B`` represents computing the derivatives of the basis functions at the quadrature points, ``J`` represents the change of variables from the grid ``G_h`` to the reference space for the element, and ``D`` represents a pointwise application of the bilinear form with quadrature weights.
With a nodal basis, the nodes on the boundary of the element are equivalent, and we can thus compute the symbol matrix as

```math
L_h = Q^T \left( A_e \odot \begin{bmatrix}
    e^{\imath \left( x_0 - x_0 \right) \theta}       && \cdots && e^{\imath \left( x_0 - x_{p + 1} \right) \theta}       \\
    \vdots                                           && \vdots && \vdots                                                 \\
    e^{\imath \left( x_{p + 1} - x_0 \right) \theta} && \cdots && e^{\imath \left( x_{p + 1} - x_{p + 1} \right) \theta} \\
\end{bmatrix} \right) Q
```

where ``\odot`` represents pointwise multiplication of the elements and

```math
Q = \begin{bmatrix}
    1       && 0      && \cdots && 0      && 1       \\
    0       && 1      && \cdots && 0      && 0       \\
    \vdots  && \vdots && \vdots && \vdots && \vdots  \\
    0       && 0      && \cdots && 1      && 0
\end{bmatrix}
```

maps the equivalent basis nodes to the same Fourier mode.

This same computation of the symbol matrix extends to more complex PDE with multiple components and in higher dimensions.

## p-Type Multigrid

Multi-grid follows the following algorithm:

1. pre-smooth: ``u_i := u_i + M^{-1} \left( b - A u_i \right)``

2. restrict: ``r_c := R \left( b - A u_i \right)``

3. coarse solve: ``A_c e_c := r_c``

4. prolongate: ``U_i := u_i + P e_c``

5. post-smooth: ``u_i := u_i + M^{-1} \left( b - A u_i \right)``

To explore the convergence of multi-grid technicques, we need to analyze the error propegation.
The spectral radius of the symbol of the error propegation operator determines how rapidly a relaxation scheme decreases error at a target frequency for a given paramenter value.
In a multi-grid technique, the purpose of the smoothing operator is to reduce the higher frequency components of the error, where low frequencies are given by ``\theta \in T^{low} = \left[ - \frac{\pi}{p}, \frac{\pi}{p} \right)`` and high frequencies are given by ``\theta \in T^{high} = \left[ - \frac{\pi}{p}, \frac{\left( 2 p - 1 \right) \pi}{p} \right) \setminus T^{low}``.

We build the symbol of the error propegation operator in parts.

### Error Relaxation Techniques

Multi-grid techniques require error relaxation techniques.
The error propagation operator for a relaxation technique is given by

```math
S = I - M^{-1} A.
```

In the specific case of Jacobi smoothing, ``M`` is given by ``M = diag \left( A \right)``.

The symbol of the error propagation operator is given by

```math
S_h \left( \omega, \theta \right) = I - M_h^{-1} L_h \left( \theta \right)
```

where ``\omega`` is a relaxation parameter.

Specifically, for Jacobi we have

```math
S_h \left( \omega, \theta \right) = I - \omega M_h^{-1} L_h \left( \theta \right)
```

where ``\omega`` is the weighting factor and ``M_h^{-1}`` is given by ``M_h^{-1} \ diag \left( L_h \right)``.

### Grid Transfer Operators



### Multigrid Operator

Combining these elements, the symbol of the error propagation operator for p-type multigrid is given by

```math
E \left( p, \theta \right) = S_f \left( p, \theta \right) \left( I - P_f \left( \theta \right) L_c^{-1} \left( p, \theta \right) R_f \left( \theta \right) L_f \left( \theta \right) \right) S_f \left( p , \theta \right).
```

## User Defined Smoothers

Example of user defined smoother. User provides element matrix and code maps to symbols.
