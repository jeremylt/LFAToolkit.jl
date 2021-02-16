## Finite Element Basis

The finite element basis objects represent the discretization of the problem.

There are convenience constructors for H1 tensor product bases on uniformly spaced, Gauss-Lobatto, or Gauss-Legendre points with Gauss-Lobatto or Gauss-Legendre quadrature.
Users can create additional finite element bases if the prerequisite information is provided.

### Base Classes

These are the bases classes for finite element bases.
The constructors for these base classes can be used to create user defined finite elements. 

```@docs
TensorBasis
NonTensorBasis
```

### Singe Element Bases

These bases represent common tensor product finite element bases used for continuous Galerkin methods.

```@docs
TensorH1LagrangeBasis
TensorH1UniformBasis
```

### Macro Element Bases

These bases represent a macro element consisting of multiple overlapping micro elements, where the micro elements are created as above.

```@docs
TensorH1LagrangeMacroBasis
TensorH1UniformMacroBasis
```

### P Prolongation Basis

This basis provides prolongation from a single element to a single element of higher order.

```@docs
TensorH1LagrangePProlongationBasis
```

### H Prolongation Bases

These bases provide prolongation from a single element to a macro element of the same order or a macro element to a macro element with a larger number of elements.

```@docs
TensorH1LagrangeHProlongationBasis
TensorH1UniformHProlongationBasis
TensorH1LagrangeHProlongationMacroBasis
TensorH1UniformHProlongationMacroBasis
```