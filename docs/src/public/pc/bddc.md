## Preconditioner: BDDC

LFAToolkit supports lumped and Dirichlet BDDC preconditioners.

```@docs
BDDCInjectionType.BDDCInjectType
```

### Lumped BDDC

#### Example

This is an example of a simple lumped BDDC preconditioner.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../../examples/ex221_lumpedbddc.jl", String))
```
""")
````

#### Documentation

```@docs
LumpedBDDC
computesymbols(::BDDC, ::Array)
```

### Dirichlet BDDC

#### Documentation

```@docs
DirichletBDDC
```
