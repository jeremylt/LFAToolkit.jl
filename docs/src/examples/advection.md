## Advection operator

This is an example of an advection operator

### Problem formulation

The advection problem is given by

```math
\dfrac{\partial \zeta}{\partial t} + U \dfrac{\partial \zeta}{\partial x} = 0
```
where ``\zeta = \zeta(x,t)`` and ``U`` is its associated phase speed.

The weak form is given by

```math
\int_\Omega v \left(\dfrac{\partial \zeta}{\partial t} \right) dV - \int_\Omega U \left( \dfrac{\partial v}{\partial x} \right) \zeta dV = 0, \forall v \in V
```

for an appropriate test space ``V \subseteq H_0^1 \left( \Omega \right)`` on the domain.