## Preconditioner: Jacobi

### Example

This is an example of a simple Jacobi smoother.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../examples/ex3_jacobi.jl", String))
```
""")
````

### Documentation

```@docs
Jacobi
computesymbols(::Jacobi, ::Array, ::Array)
```
