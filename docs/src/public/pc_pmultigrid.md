## Preconditioner: P-Multigrid

### Examples

This is an example of a simple p-multigrid preconditioner.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../examples/ex201_pmultigrid.jl", String))
```
""")
````

This is an example of a multilevel p-multigrid preconditioner.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../examples/ex202_pmultigrid_multilevel.jl", String))
```
""")
````

### Documentation

```@docs
PMultigrid
computesymbols(::PMultigrid, ::Array, ::Array{Int}, ::Array)
```
