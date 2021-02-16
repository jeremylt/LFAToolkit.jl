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

#### Documentation

```@docs
PMultigrid
computesymbols(::Multigrid, ::Array, ::Array{Int}, ::Array)
```

### H-Multigrid

#### Example

This is an example of a simple h-multigrid preconditioner.

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

#### Documentation

```@docs
HMultigrid
```