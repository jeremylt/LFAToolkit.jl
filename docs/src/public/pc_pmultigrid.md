## Preconditioner: P-Multigrid

This is an example of a simple p-multigrid preconditioner.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../examples/ex4_pmultigrid.jl", String))
```
""")
````

```@docs
PMultigrid
computesymbols(::PMultigrid, ::Array, ::Array{Int}, ::Array)
```
