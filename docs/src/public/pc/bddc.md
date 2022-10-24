## Preconditioner: BDDC

LFAToolkit supports lumped and Dirichlet BDDC preconditioners.

```@docs
BDDCInjectionType.BDDCInjectType
```

### Lumped BDDC

#### Examples

This is an example of a simple lumped BDDC preconditioner.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../../examples/ex221_lumped_bddc.jl", String))
```
""")
````
![](../../img/221_lumped_bddc_spectral_radius_3_2d.png)

This is an example of a simple lumped BDDC preconditioner on a macro-element patch.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../../examples/ex222_lumped_bddc.jl", String))
```
""")
````
![](../../img/222_lumped_bddc_spectral_radius_4_elem_2d.png)

#### Documentation

```@docs
LumpedBDDC
computesymbols(::BDDC, ::Array{<:Real}, ::Array{<:Real})
```

### Dirichlet BDDC

#### Examples

This is an example of a simple Dirichlet BDDC preconditioner.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../../examples/ex223_dirichlet_bddc.jl", String))
```
""")
````
![](../../img/223_dirichlet_bddc_spectral_radius_3_2d.png)

This is an example of a simple Dirichlet BDDC preconditioner on a macro-element patch.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../../examples/ex224_dirichlet_bddc.jl", String))
```
""")
````
![](../../img/224_dirichlet_bddc_spectral_radius_4_elem_2d.png)

#### Documentation

```@docs
DirichletBDDC
```
