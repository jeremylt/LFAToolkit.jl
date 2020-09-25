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
function Base.show(io::IO, preconditioner::Jacobi)
    print(io, "jacobi preconditioner")
end

# ---------------------------------------------------------------------------------------------------------------------
# data for computing symbols
# ---------------------------------------------------------------------------------------------------------------------

function getoperatordiagonalinverse(preconditioner::Jacobi)
    # assemble if needed
    if !isdefined(preconditioner, :operatordiagonalinverse)
        # retrieve and invert
        diagonal = preconditioner.operator.diagonal
        diagonalinverse = [abs(val) > 1e-15 ? 1.0 / val : 0.0 for val in diagonal]

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
computesymbols()
```

Compute or retrieve the symbol matrix for a Jacobi preconditioned operator

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

# note: either syntax works
A = computesymbols(jacobi, π);

# verify
using LinearAlgebra;
eigenvalues = eigvals(A);
@assert max(real(eigenvalues)...) ≈ 1.0
 
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
A = computesymbols(jacobi, π, π);

# verify
using LinearAlgebra;
eigenvalues = eigvals(A);
@assert max(real(eigenvalues)...) ≈ 5/4
 
# output

```
"""
function computesymbols(preconditioner::Jacobi, θ::Array)
    # return
    return preconditioner.operatordiagonalinverse *
           computesymbols(preconditioner.operator, θ)
end

function computesymbols(preconditioner::Jacobi, θ_x::Number)
    # return
    return computesymbols(preconditioner, [θ_x])
end

function computesymbols(preconditioner::Jacobi, θ_x::Number, θ_y::Number)
    # return
    return computesymbols(preconditioner, [θ_x, θ_y])
end

function computesymbols(preconditioner::Jacobi, θ_x::Number, θ_y::Number, θ_z::Number)
    # return
    return computesymbols(preconditioner, [θ_x, θ_y, θ_z])
end

# ---------------------------------------------------------------------------------------------------------------------
