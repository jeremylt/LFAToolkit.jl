# Mathematical Background

Local Fourier Analysis was first used by Brandt to analyze the convergence of multi-level adaptive techniques, but the technique has been adapted for multi-level and multi-grid techniques more broadly.


By way of example, we will explore Local Fourier Analysis with the diffusion operator.


Consider the PDE

```math
- \nabla^2 u = f
```

with corresponding weak form

```math
\int_{\Omega} \nabla u \cdot \nabla v - \int_{\partial \Omega} \nabla u v = \int_{\Omega} f v, \forall v \in V
```

for some suitable ``V \subseteq H_0^1 \left( \Omega \right)``.

In Local Fourier Analysis, we focus on single elements or macro-element patches, neglecting the boundary conditions by assuming the boundary is distant from the local element under consideration.

```math
a \left( u, v \right) = \int_{\Omega} \nabla u \cdot \nabla v
```

In the specific case of a one dimensional mesh with cubic Lagrage basis on the Gauss-Lobatto points, the assembled stiffness matrix is given by
INSERT STIFFNESS MATRIX HERE


