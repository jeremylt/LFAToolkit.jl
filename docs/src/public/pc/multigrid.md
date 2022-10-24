## Preconditioner: Multigrid

LFAToolkit supports both p-multigrid and h-multigrid.

```@docs
MultigridType.MgridType
```

### P-Multigrid

#### Example

This is an example of a simple p-multigrid preconditioner.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../../examples/ex201_pmultigrid.jl", String))
```
""")
````

![](../../img/201_pmultigrid_spectral_radius_2_to_1_2d.png)

Example plot for the symbol of p-multigrid with a cubic Chebyshev smoother for the 2D scalar diffusion problem with cubic basis.

This is an example of a multilevel p-multigrid preconditioner.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../../examples/ex202_pmultigrid_multilevel.jl", String))
```
""")
````

![](../../img/202_pmultigrid_spectral_radius_4_to_2_to_1_2d.png)

Example plot for the symbol of multilevel p-multigrid with a cubic Chebyshev smoother for the 2D scalar diffusion problem with cubic basis.

#### Documentation

```@docs
PMultigrid
computesymbols(::Multigrid, ::Array{<:Real}, ::Array{Int}, ::Array{<:Real})
```

### H-Multigrid

#### Example

This is an example of a simple h-multigrid preconditioner.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../../examples/ex211_hmultigrid.jl", String))
```
""")
````

![](../../img/211_hmultigrid_spectral_radius_2_to_1_2d.png)

Example plot for the symbol of h-multigrid with a cubic Chebyshev smoother for the 2D scalar diffusion problem with linear basis.

This is an example of a multilevel h-multigrid preconditioner.

````@eval
using Markdown 
Markdown.parse("""
```julia
$(read("../../../../examples/ex212_hmultigrid_multilevel.jl", String))
```
""")
````

![](../../img/212_hmultigrid_spectral_radius_4_to_2_to_1_2d.png)

Example plot for the symbol of multilevel h-multigrid with a cubic Chebyshev smoother for the 2D scalar diffusion problem with linear basis.

#### Documentation

```@docs
HMultigrid
```
