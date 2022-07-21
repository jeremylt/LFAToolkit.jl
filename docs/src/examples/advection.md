## Advection operator

This is an example of an advection operator by using polynomial bases vs nonpolynomial bases
The choices of nonpolynomial bases available are:
mapping = sausage_transformation(9)
mapping = kosloff_talezer_transformation(0.98)
mapping = hale_trefethen_strip_transformation(1.4)

### Problem formulation

The advection problem is given by

```math
\dfrac{\partial u}{\partial t} + c \dfrac{\partial u}{\partial x} = 0
```
where ``u = u(x,t)`` and ``c`` is its associated phase speed.

The weak form is given by

```math
\int_\Omega v \left(\dfrac{\partial u}{\partial t} \right) dV - \int_\Omega c u \left( \dfrac{\partial v}{\partial x} \right) dV = 0, \forall v \in V
```

for an appropriate test space ``V \subseteq H^1 \left( \Omega \right)`` on the domain.
In this weak formulation, boundary terms have been omitted, as they are not present on the infinite grid for Local Fourier Analysis.

For understanding about nonpolynomial bases, see paper Hale and Trefethen (2008) New quadrature formulas from conformal maps. https://doi.org/10.1137/07068607X

### LFAToolkit code

The advection operator is a classical test case to see dispersion spectrum inside LFAToolkit.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../examples/ex010_advection.jl", String))
```
""")
````