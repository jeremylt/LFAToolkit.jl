# ---------------------------------------------------------------------------------------------------------------------
# Jacobi preconditioner
# ---------------------------------------------------------------------------------------------------------------------

"""
```julia
Jacobi(operator)
```

Jacobi diagonal preconditioner for finite element operators

# Arguments:
- `operator`: finite element operator to precondition

# Returns:
- Jacobi preconditioner object

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
basis = TensorH1LagrangeBasis(4, 4, 2);
    
function massweakform(u::Array{Float64}, w::Array{Float64})
    v = u * w[1]
    return [v]
end
    
# mass operator
inputs = [
    OperatorField(basis, [EvaluationMode.interpolation]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.interpolation])];
mass = Operator(massweakform, mesh, inputs, outputs);

# preconditioner
jacobi = Jacobi(mass);

# verify
println(jacobi)
println(jacobi.operator)

# output
jacobi preconditioner
finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    dimension: 2
  evaluation mode:
    interpolation
operator field:
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    dimension: 2
  evaluation mode:
    interpolation
```
"""
mutable struct Jacobi
    # data never changed
    operator::Operator

    # data empty until assembled
    operatordiagonalinverse::AbstractArray{Float64}

    # inner constructor
    Jacobi(operator) = new(operator)
end

# printing
# COV_EXCL_START
Base.show(io::IO, preconditioner::Jacobi) = print(io, "jacobi preconditioner")
# COV_EXCL_STOP

# ---------------------------------------------------------------------------------------------------------------------
# data for computing symbols
# ---------------------------------------------------------------------------------------------------------------------

"""
```julia
getoperatordiagonalinverse(preconditioner)
```

Compute or retrieve the inverse of the symbol matrix diagonal for a Jacobi preconditioner

# Returns:
- Symbol matrix diagonal inverse for the operator

# Example:
```jldoctest
# setup
mesh = Mesh1D(1.0);
basis = TensorH1LagrangeBasis(3, 4, 1);
    
function diffusionweakform(du::Array{Float64}, w::Array{Float64})
    dv = du * w[1]
    return [dv]
end
    
# mass operator
inputs = [
    OperatorField(basis, [EvaluationMode.gradient]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.gradient])];
diffusion = Operator(diffusionweakform, mesh, inputs, outputs);

# preconditioner
jacobi = Jacobi(diffusion)

# note: either syntax works
diagonalinverse = LFAToolkit.getoperatordiagonalinverse(jacobi);
diagonalinverse = jacobi.operatordiagonalinverse;

# verify
@assert diagonalinverse ≈ [6/7 0; 0 3/4]
 
# output

```
"""
function getoperatordiagonalinverse(preconditioner::Jacobi)
    # assemble if needed
    if !isdefined(preconditioner, :operatordiagonalinverse)
        # retrieve diagonal and invert
        diagonalinverse = preconditioner.operator.diagonal^-1

        # store
        preconditioner.operatordiagonalinverse = diagonalinverse
    end

    # return
    return getfield(preconditioner, :operatordiagonalinverse)
end

# ---------------------------------------------------------------------------------------------------------------------
# get/set property
# ---------------------------------------------------------------------------------------------------------------------

function Base.getproperty(preconditioner::Jacobi, f::Symbol)
    if f == :operatordiagonalinverse
        return getoperatordiagonalinverse(preconditioner)
    else
        return getfield(preconditioner, f)
    end
end

function Base.setproperty!(preconditioner::Jacobi, f::Symbol, value)
    if f == :operator
        throw(ReadOnlyMemoryError()) # COV_EXCL_LINE
    else
        return setfield!(preconditioner, f, value)
    end
end

# ---------------------------------------------------------------------------------------------------------------------
# compute symbols
# ---------------------------------------------------------------------------------------------------------------------

"""
```julia
computesymbols(preconditioner, ω, θ_x, ...)
```

Compute or retrieve the symbol matrix for a Jacobi preconditioned operator

# Arguments:
- `preconditioner`: Jacobi preconditioner to compute symbol matrix for
- `ω`:              Smoothing weighting factor
- `θ_x`:            Fourier mode frequency in x direction
- `θ_y`:            Fourier mode frequency in y direction (2D and 3D)
- `θ_z`:            Fourier mode frequency in z direction (3D)

# Returns:
- Symbol matrix for the Jacobi preconditioned operator

# 1D Example:
```jldoctest
# setup
mesh = Mesh1D(1.0);
basis = TensorH1LagrangeBasis(3, 4, 1);
    
function diffusionweakform(du::Array{Float64}, w::Array{Float64})
    dv = du * w[1]
    return [dv]
end
    
# mass operator
inputs = [
    OperatorField(basis, [EvaluationMode.gradient]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.gradient])];
diffusion = Operator(diffusionweakform, mesh, inputs, outputs);

# preconditioner
jacobi = Jacobi(diffusion);

# compute symbols
A = computesymbols(jacobi, 1.0, π);

# verify
using LinearAlgebra;
eigenvalues = eigvals(A);
@assert real(eigenvalues) ≈ [0; 1/7]
 
# output

```
# 2D Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
basis = TensorH1LagrangeBasis(3, 4, 2);
    
function diffusionweakform(du::Array{Float64}, w::Array{Float64})
    dv = du * w[1]
    return [dv]
end
    
# mass operator
inputs = [
    OperatorField(basis, [EvaluationMode.gradient]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.gradient])];
diffusion = Operator(diffusionweakform, mesh, inputs, outputs);

# preconditioner
jacobi = Jacobi(diffusion)

# compute symbols
A = computesymbols(jacobi, 1.0, π, π);

# verify
using LinearAlgebra;
eigenvalues = eigvals(A);
@assert real(eigenvalues) ≈ [-1/4; -1/14; 0; 1/7]
 
# output

```
"""
function computesymbols(preconditioner::Jacobi, ω::Number, θ::Array)
    # return
    S_A = preconditioner.operatordiagonalinverse * computesymbols(preconditioner.operator, θ...)
    return I - ω * S_A
end

function computesymbols(preconditioner::Jacobi, ω::Number, θ_x::Number)
    return computesymbols(preconditioner, ω, [θ_x])
end

function computesymbols(preconditioner::Jacobi, ω::Number, θ_x::Number, θ_y::Number)
    return computesymbols(preconditioner, ω, [θ_x, θ_y])
end

function computesymbols(
    preconditioner::Jacobi,
    ω::Number,
    θ_x::Number,
    θ_y::Number,
    θ_z::Number,
)
    return computesymbols(preconditioner, ω, [θ_x, θ_y, θ_z])
end

# ---------------------------------------------------------------------------------------------------------------------
