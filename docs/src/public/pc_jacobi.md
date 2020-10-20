## Preconditioner: Jacobi

This is an example of a simple Jacobi smoother.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../test/ex3_jacobi.jl", String))
```
""")
````

```@docs
Jacobi
computesymbols(::Jacobi, ::Array, ::Array)
```
