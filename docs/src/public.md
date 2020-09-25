# Public API

This page documents the public API of the LFAToolkit.

## Mesh

```@docs
Mesh1D
Mesh2D
Mesh3D
```

## Basis

```@docs
TensorBasis
NonTensorBasis
TensorH1LagrangeBasis
```

## Basis Evaluation Mode

```@docs
EvaluationMode.EvalMode
```

## Operator Field

```@docs
OperatorField
```

## Operator

```@docs
Operator
computesymbols(::Operator, ::Array)
```

## Preconditioners

```@docs
Jacobi
computesymbols(::Jacobi, ::Array)
```
