# Mathematical Background

Local Fourier Analysis was first used by Brandt [1] to analyze the convergence of multi-level adaptive techniques for solving PDEs discretized with finite differences, but the technique has been adapted for multi-level and multi-grid techniques using finite element discretizations.

By way of example, we will explore Local Fourier Analysis for multilevel and mutligrid methods for the two dimensional Poisson problem, given by the PDE

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
L_h \varphi \left( \theta, x \right) = \tilde{L}_h \left( \theta \right) \varphi \left( \theta, x \right)
```

then ``\tilde{L}_h \left( \theta \right) = \sum_{\kappa \in V} s_\kappa e^{\imath \theta \kappa}`` is the **symbol** of ``L_h``.

We can extend this to a ``p \times p`` linear system of operators representing a scalar problem on a ``p`` order finite element

```math
\tilde{L}_h = \begin{bmatrix}
    \tilde{L}_h^{1, 1}  &&  \cdots  &&  \tilde{L}_h^{1, p} \\
    \vdots              &&  \vdots  &&  \vdots             \\
    \tilde{L}_h^{p, 1}  &&  \cdots  &&  \tilde{L}_h^{p, p} \\
\end{bmatrix}
```

where ``\tilde{L}_h^{i, j}`` is given by a scalar Toeplitz operator describing how component ``j`` appears in the equation for component ``i``.

Consider the specific case of the Topeliz operator representing the scalar diffusion operator.
The Poisson problem has the weak formulation

```math
\int_{\Omega} \nabla u \cdot \nabla v - \int_{\partial \Omega} \nabla u v = \int_{\Omega} f v, \forall v \in V
```

for some suitable ``V \subseteq H_0^1 \left( \Omega \right)``.

Selecting a finite element basis, we can discretize the weak form and produce

```math
A u = b
```

Using the algebraic representation of PDE operators discussed in [2], the assembled matrix is of the form

```math
A = P^T A_e P
```

```math
A_e = B^T J^T D J B
```

where ``P`` represents the element assembly operator, ``B`` represents computing the derivatives of the basis functions at the quadrature points, ``J`` represents the change of variables from the grid ``G_h`` to the reference space for the element, and ``D`` represents a pointwise application of the bilinear form with quadrature weights.
As we are on the infinite grid, ``G_h``, boundary conditions have been omitted.
This analysis will also be equivalent to periodic boundary conditions.

With a nodal basis of order ``p``, the nodes on the boundary of the element are equivalent, and we can thus compute the symbol matrix as

```math
\tilde{A}_h = Q^T \left( A_e \odot \begin{bmatrix}
    e^{\imath \left( x_0 - x_0 \right) \theta}        &&  \cdots  &&  e^{\imath \left( x_0 - x_{p + 1} \right) \theta}        \\
    \vdots                                            &&  \vdots  &&  \vdots                                                  \\
    e^{\imath \left( x_{p + 1} - x_0 \right) \theta}  &&  \cdots  &&  e^{\imath \left( x_{p + 1} - x_{p + 1} \right) \theta}  \\
\end{bmatrix} \right) Q
```

where ``\odot`` represents pointwise multiplication of the elements and

```math
Q = \begin{bmatrix}
    1       &&  0       &&  \cdots  &&  0       &&  1       \\
    0       &&  1       &&  \cdots  &&  0       &&  0       \\
    \vdots  &&  \vdots  &&  \vdots  &&  \vdots  &&  \vdots  \\
    0       &&  0       &&  \cdots  &&  1       &&  0       \\
\end{bmatrix}
```

maps the equivalent basis nodes to the same Fourier mode.

This same computation of the symbol matrix extends to more complex PDE with multiple components and in higher dimensions.

## Multigrid

Multigrid follows the following algorithm:

1. pre-smooth   : ``u_i := u_i + M^{-1} \left( b - A u_i \right)``

2. restrict     : ``r_c := R_{ftoc} \left( b - A u_i \right)``

3. coarse solve : ``A_c e_c := r_c``

4. prolongate   : ``u_i := u_i + P_{ctof} e_c``

5. post-smooth  : ``u_i := u_i + M^{-1} \left( b - A u_i \right)``

where ``f`` and ``c`` represent the fine and coarse grids, respectively, ``R_{ftoc}`` represents the grid restriction operator, ``P_{ctof}`` represents the grid prolongation operator.

To explore the convergence of multigrid techniques, we need to analyze the symbol of the multigrid error propagation operator

```math
E_f \left( p, \theta \right) = S_h \left( p, \theta \right) E_c \left( \theta \right) S_h \left( p, \theta \right).
```

The symbol of the coarse grid error propagation operator is given by

```math
E_c \left( \theta \right) = I - \tilde{P}_{ctof} \left( \theta \right) \tilde{A}_c^{-1} \left( \theta \right) \tilde{R}_{ftoc} \left( \theta \right) \tilde{A}_f \left( \theta \right).
```

The spectral radius of the symbol of the error propagation operator determines how rapidly a relaxation scheme decreases error at a target frequency for a given parameter value.
In a multigrid technique, the purpose of the smoothing operator is to reduce the higher frequency components of the error, where low frequencies are given by ``\theta \in T^{low} = \left[ - \frac{\pi}{p}, \frac{\pi}{p} \right)`` and high frequencies are given by ``\theta \in T^{high} = \left[ - \frac{\pi}{p}, \frac{\left( 2 p - 1 \right) \pi}{p} \right) \setminus T^{low}``.

We build the symbol of the multigrid error propagation operator in parts.

### Smoothing Operator

Multigrid techniques require error relaxation techniques.
The error propagation operator for a relaxation technique is given by

```math
S = I - M^{-1} A.
```

In the specific case of Jacobi smoothing, ``M`` is given by ``M = diag \left( A \right)``.

The symbol of the error propagation operator is given by

```math
S_h \left( \omega, \theta \right) = I - M_h^{-1} \tilde{A}_h \left( \theta \right)
```

where ``\omega`` is a relaxation parameter.

Specifically, for Jacobi we have

```math
S_h \left( \omega, \theta \right) = I - \omega M_h^{-1} \tilde{A}_h \left( \theta \right)
```

where ``\omega`` is the weighting factor and ``M_h`` is given by

```math
M_h = Q^T \left( diag \left( A_e \right) \odot \begin{bmatrix}
    e^{\imath \left( x_0 - x_0 \right) \theta}        &&  \cdots  &&  e^{\imath \left( x_0 - x_{p + 1} \right) \theta}        \\
    \vdots                                            &&  \vdots  &&  \vdots                                                  \\
    e^{\imath \left( x_{p + 1} - x_0 \right) \theta}  &&  \cdots  &&  e^{\imath \left( x_{p + 1} - x_{p + 1} \right) \theta}  \\
\end{bmatrix} \right) Q.
```

If multiple pre or post-smoothing passes are used, we have

```math
S_h \left( \omega, \nu, \theta \right) = \left( I - \omega M_h^{-1} \tilde{A}_h \left( \theta \right) \right)^{\nu}
```

where ``\nu`` is the number of smoothing passes.

### Grid Transfer Operators

We consider grid transfer operators for p-type multigrid.
Prolongation from the lower order coarse grid to the high order fine grid is given by 

```math
P_{ctof} = P_{fine}^T D_{scale} B_{c to f} P_{coarse}
```

where ``B_{c to f}`` is a basis interpolation from the coarse basis to the fine basis, ``P_{fine}`` is the fine grid element assembly operator, ``P_{coarse}`` is the coarse grid element assembly operator, and ``D_{scale}`` is a scaling for node multiplicity across elements.

Restriction from the fine grid to the coarse grid is given by the transpose, ``R_{ftoc} = P{ctof}^T``.

Thus, the symbol of ``P_{ctof}`` is given by

```math
\tilde{P}_{ctof} = diag \left( Q_f \right)^T \left( B_{ctof} \odot \begin{bmatrix}
    e^{\imath \left( x_{0, f} - x_{0, c} \right) \theta}          &&  \cdots  &&  e^{\imath \left( x_{0, f} - x_{p_c + 1, c} \right) \theta}        \\
    \vdots                                                        &&  \vdots  &&  \vdots                                                            \\
    e^{\imath \left( x_{p_f + 1, f} - x_{0, c} \right) \theta}    &&  \cdots  &&  e^{\imath \left( x_{p_f + 1, f} - x_{p_c + 1, c} \right) \theta}  \\
\end{bmatrix} \right) diag \left( Q_c \right)
```

and ``\tilde{R}_{ftoc}`` is given by the transpose.

### Multigrid Error Propagation Operator

Combining these elements, the symbol of the error propagation operator for p-type multigrid is given by

```math
E \left( p, \theta \right) = S_f \left( p, \theta \right) \left[ I - \tilde{P}_{ctof} \left( \theta \right) \tilde{A}_c^{-1} \left( p, \theta \right) \tilde{R}_{ftoc} \left( \theta \right) \tilde{A}_f \left( \theta \right) \right] S_f \left( p , \theta \right)
```

where ``\tilde{P}_{ctof}`` and ``\tilde{R}_{ftoc}`` are given above, ``S_h`` is given by the smoothing operator, and ``\tilde{A}_c`` and ``\tilde{A}_f`` are derived from the PDE being analyzed.

## User Defined Smoothers

ToDo: Example of user defined smoother. User provides element matrix and code maps to symbols.
