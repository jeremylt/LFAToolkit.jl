## Neo-Hookean hyperelasticity

This is an example of a p-multigrid preconditioner with Chebyshev smoothing for the 3D Neo-Hookean hyperelasticity problem.

### Problem formulation

The strong form of the static balance of linear-momentum at finite strain is given by

```math
-\nabla_x \cdot \bold{P} - \rho_0 \bold{g} = \bold{0}
```

where ``-\nabla_k`` is the gradient with respect to the reference configuration, ``\bold{P}`` is the first Piola-Kirchhoff stress tensor, ``\rho_0`` is the reference mass density, and ``\bold{g}`` is the forcing function.

The first Piola-Kirchhoff stress tensor is given by

```math
\bold{P} = \bold{F} \bold{S}
```

where ``\bold{F}`` is the deformation gradient and ``\bold{S}`` is the second Piola-Kirchhoff stress tensor.
In this example, the second Piola-Kirchhoff stress tensor is given by the Neo-Hookean model.

For a full discussion of the formulation of the 3D Neo-Hookean hyperelasticity problem, see the [libCEED documentation](https://libceed.readthedocs.io/en/latest/examples/solids/).

### LFAToolkit code

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../examples/ex311_hyperelasticity.jl", String))
```
""")
````
