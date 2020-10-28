## Finite Element Operator

### Examples

This is an example of a simple mass operator in two dimensions.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../examples/ex1_mass.jl", String))
```
""")
````

This is an example of a simple diffusion operator in two dimensions.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../examples/ex2_diffusion.jl", String))
```
""")
````

### Documentation

```@docs
Operator
GalleryOperator
computesymbols(::Operator, ::Array)
```
