## Preconditioner: Jacobi

This smoother provides Jacobi smoothing based on the operator diagonal.

### Example

This is an example of a simple Jacobi smoother.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../examples/ex101_jacobi.jl", String))
```
""")
````

### Documentation

```@docs
Jacobi
computesymbols(::Jacobi, ::Array, ::Array)
```