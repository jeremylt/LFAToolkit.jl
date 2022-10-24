## Preconditioner: Jacobi

This smoother provides Jacobi smoothing based on the operator diagonal.

### Example

This is an example of a simple Jacobi smoother.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../../examples/ex101_jacobi.jl", String))
```
""")
````

![](../../img/101_jacobi_spectral_radius_2_2d.png)

Plot for the symbol of a Jacobi smoother for the 2D scalar diffusion problem with cubic basis.

### Documentation

```@docs
Jacobi
computesymbols(::Jacobi, ::Array{<:Real}, ::Array{<:Real})
```
