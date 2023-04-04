## Preconditioner: Chebyshev

This smoother provides Chekyshev polynomial smoothing of a runtime specified order.
See [lottes2022](@cite) and [phillips2022](@cite) for discussion of Chebyshev smoother types for multigrid V-cycles.

### Example

This is an example of a Chebyshev smoother.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../../examples/ex111_chebyshev.jl", String))
```
""")
````

![](../../img/111_chebyshev_spectral_radius_2_2d.png)

Plot for the symbol a cubic Chebyshev smoother for the 2D scalar diffusion problem with cubic basis.

### Documentation

```@docs
Chebyshev
seteigenvalueestimatescaling
computesymbols(::Chebyshev, ::Array{<:Real}, ::Array{<:Real})
```
