## Linear elasticity

This is an example of a p-multigrid preconditioner with Chebyshev smoothing for the 3D linear elasticity problem.

### Problem formulation

The strong form of the static balance of linear-momentum at finite strain is given by

```math
\nabla \cdot \boldsymbol{\sigma} + \bold{g} = \bold{0}
```

where ``\boldsymbol{\sigma}`` and ``\bold{g}`` are the stress and forcing functions, respectively.

We have the weak form

```math
\int_{\Omega} \nabla \bold{v} : \boldsymbol{\sigma} - \int_{\partial \Omega} v \cdot \left( \boldsymbol{\sigma} \cdot \hat{\bold{n}} \right) - \int_{\Omega} \bold{v} \cdot \bold{g} = 0.
```

The constitutive law (stress-strain relationship) can be written as

```math
\boldsymbol{\sigma} = \bold{C} : \boldsymbol{\epsilon}
```

where ``\boldsymbol{\epsilon}`` is the infinitesimal strain tensor

```math
\boldsymbol{\epsilon} = \frac{1}{2} \left( \nabla \bold{u} + \nabla \bold{u}^T \right)
```

and the elasticity tensor ``\bold{C}`` is given by

```math
\bold{C} =
\begin{pmatrix}
   \lambda + 2\mu & \lambda & \lambda & & & \\
   \lambda & \lambda + 2\mu & \lambda & & & \\
   \lambda & \lambda & \lambda + 2\mu & & & \\
   & & & \mu & & \\
   & & & & \mu & \\
   & & & & & \mu
   \end{pmatrix}.
```

``\lambda`` and ``\mu`` are the Lamé parameters, given by

```math
\lambda = \frac{E \nu}{\left( 1 + \nu \right) \left( 1 - 2 \nu \right)} \\
\mu = \frac{E}{2 \left( 1 + \nu \right)}
```
where ``E`` is the Young's modulus and ``\nu`` is the Poisson's ratio for the materiel.

For a full discussion of the formulation of the linear elasticity problem, see the [libCEED documentation](https://libceed.readthedocs.io/en/latest/examples/solids/).

### LFAToolkit code

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../examples/ex301_linear_elasticity.jl", String))
```
""")
````
