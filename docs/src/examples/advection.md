## Advection operator

This is an example of an advection operator by using polynomial bases vs nonpolynomial bases
The choices of nonpolynomial bases available are:
mapping = sausagetransformation(9)
mapping = koslofftalezertransformation(0.98)
mapping = haletrefethenstriptransformation(1.4)

### Problem formulation

The advection problem is given by

```math
\dfrac{\partial u}{\partial t} + c \dfrac{\partial u}{\partial x} = 0
```
where ``\zeta = \zeta(x,t)`` and ``U`` is its associated phase speed.

The weak form is given by

```math
\int_\Omega v \left(\dfrac{\partial \zeta}{\partial t} \right) dV - \int_\Omega U \zeta \left( \dfrac{\partial v}{\partial x} \right) dV = 0, \forall v \in V
```

for an appropriate test space ``V \subseteq H^1 \left( \Omega \right)`` on the domain.
In this weak formulation, boundary terms have been omitted, as they are not present on the infinite grid for Local Fourier Analysis.

For understanding about nonpolynomial bases, see paper New quadrature formulas from conformal maps
SIAM J. NUMER. ANAL. Vol. 46, No. 2 (2008) by following Hale and Trefethen (2008)

### LFAToolkit code

The advection operator is a classical test case to see dipersion spectrum inside LFAToolkit.