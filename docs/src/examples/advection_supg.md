## Streamline-Upwind/Petrov-Galerkin (SUPG) advection operator

This is an example of the 1D scalar SUPG advection problem.

### Problem formulation

The advection problem is given by

```math
\dfrac{\partial u}{\partial t} + c \dfrac{\partial u}{\partial x} = 0
```
where ``u = u(x,t)`` and ``c`` is its associated phase speed.

The SUPG advection weak form is given by

```math
\int_\Omega v \left(\dfrac{\partial u}{\partial t} \right) dV - \int_\Omega \dfrac{\partial v}{\partial x} \left(c u - c τ \left( \dfrac{\partial u}{\partial t} + c \dfrac{\partial u}{\partial x} \right) \right) dV = 0, \forall v \in V
```
for an appropriate test space ``V \subseteq H^1 \left( \Omega \right)`` on the domain.
In this weak formulation, boundary terms have been omitted, as they are not present on the infinite grid for Local Fourier Analysis.
The SUPG stabilization is controlled by the parameter ``τ``, where ``τ = 0`` gives the classical Galerkin formulation and ``τ = \dfrac{h}{2}`` gives a nodally exact solution to the steady advection equation with source when using linear elements (this can be extended to advection-diffusion with a further scaling that depends on the cell Péclet number).

For discussion on SUPG, see [hughes1979](@cite), [brooks1982](@cite), and [whiting2003](@cite).

### LFAToolkit code

The symbol of the continuous advection operator ``u_t + c u_x = 0`` applied to the Fourier mode ``e^{i\theta x}`` is ``i\theta``.
The finite element discretization yields ``M u_t + A u = 0``, and thus we are interested in the symbol of ``-M^{-1} A``.
One may compare the continuous spectrum with the discrete symbol, which is necessarily periodic on ``[-\pi, \pi)``, to understand the behavior for high wave numbers, including the high frequencies that will limit stable time steps.
To understand dispersion within the resolved frequencies, we instead plot the phase speed ``\lambda/\theta``, which should be very close to ``c`` through the resolved frequencies.
Here we show the SUPG advection operator on ``H^1`` Lagrange basis.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../examples/ex011_advection_supg.jl", String))
```
""")
````