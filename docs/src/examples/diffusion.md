## Diffusion operator

This is an example of various preconditioners for the 2D scalar diffusion problem.

### Problem formulation

The scalar diffusion problem is given by

```math
\nabla^2 u = f
```

with a weak form of

```math
\int_\Omega \nabla u \nabla v = \int_\Omega v f, \forall v \in V
```

for an appropriate test space ``V \subseteq H_0^1 \left( \Omega \right)`` on the domain.
In this weak formulation, boundary terms have been omitted, as they are not present on the infinite grid for Local Fourier Analysis.

### LFAToolkit code

This is an example of a Chebyshev smoother.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../examples/ex111_chebyshev.jl", String))
```
""")
````

This is an example of a two level p-multigrid preconditioner.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../examples/ex201_pmultigrid.jl", String))
```
""")
````

![](../img/multi_grid_spectral_radius_9_to_2_2d.png)

Example plot for the symbol of two level p-multigrid with a Jacobi smoother for the 2D scalar diffusion problem.

This is an example of a multilevel p-multigrid preconditioner.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../examples/ex202_pmultigrid_multilevel.jl", String))
```
""")
````

This is an example of a two level h-multigrid preconditioner.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../examples/ex211_hmultigrid.jl", String))
```
""")
````

This is an example of a multilevel h-multigrid preconditioner.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../examples/ex212_hmultigrid_multilevel.jl", String))
```
""")
````
