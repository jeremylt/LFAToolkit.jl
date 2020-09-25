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

    # data may change
    operatordiagonal::AbstractArray{Float64}

    # constructor
    Jacobi(operator) = new(operator)
end


# printing
function Base.show(io::IO, preconditioner::Jacobi)
    print(io, "jacobi preconditioner")
end

# ---------------------------------------------------------------------------------------------------------------------
# get/set property
# ---------------------------------------------------------------------------------------------------------------------

function Base.setproperty!(preconditioner::Jacobi, f::Symbol, value)
    if f == :operator
        throw(ReadOnlyMemoryError()) # COV_EXCL_LINE
    else
        return setfield!(preconditioner, f, value)
    end
end

# ---------------------------------------------------------------------------------------------------------------------
