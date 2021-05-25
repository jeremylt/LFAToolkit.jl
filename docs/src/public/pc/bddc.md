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
![](../../img/221_lumpedbddc_spectral_radius_4_2d.png)

This is an example of a simple lumped BDDC preconditioner on a macro-element patch.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../../examples/ex222_lumpedbddc.jl", String))
```
""")
````
![](../../img/222_lumpedbddc_spectral_radius_4_elem_2d.png)

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
