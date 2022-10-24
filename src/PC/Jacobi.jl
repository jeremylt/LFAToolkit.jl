# ------------------------------------------------------------------------------
# Jacobi preconditioner
# ------------------------------------------------------------------------------

"""
```julia
Jacobi(operator)
```

Jacobi diagonal preconditioner for finite element operators

# Arguments:

  - `operator::Operator`:  finite element operator to precondition

# Returns:

  - Jacobi preconditioner object

# Example:

```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
mass = GalleryOperator("mass", 4, 4, mesh);

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
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    interpolation
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    interpolation
```
"""
mutable struct Jacobi <: AbstractPreconditioner
    # data never changed
    operator::Operator

    # data empty until assembled
    operatordiagonalinverse::AbstractArray{Float64}

    # inner constructor
    Jacobi(operator::Operator) = new(operator)
end

# printing
# COV_EXCL_START
Base.show(io::IO, _::Jacobi) = print(io, "jacobi preconditioner")
# COV_EXCL_STOP

# ------------------------------------------------------------------------------
# data for computing symbols
# ------------------------------------------------------------------------------

"""
```julia
getoperatordiagonalinverse(preconditioner)
```

Compute or retrieve the inverse of the symbol matrix diagonal for a Jacobi

# Returns:

  - symbol matrix diagonal inverse for the operator

# Example:

```jldoctest
# setup
mesh = Mesh1D(1.0);
diffusion = GalleryOperator("diffusion", 3, 3, mesh);

# preconditioner
jacobi = Jacobi(diffusion);

# verify operator diagonal inverse
@assert jacobi.operatordiagonalinverse ≈ [6/7 0; 0 3/4]

# output

```
"""
function getoperatordiagonalinverse(jacobi::Jacobi)
    # assemble if needed
    if !isdefined(jacobi, :operatordiagonalinverse)
        # retrieve diagonal and invert
        diagonalinverse = jacobi.operator.diagonal^-1

        # store
        jacobi.operatordiagonalinverse = diagonalinverse
    end

    # return
    return getfield(jacobi, :operatordiagonalinverse)
end

# ------------------------------------------------------------------------------
# get/set property
# ------------------------------------------------------------------------------

function Base.getproperty(jacobi::Jacobi, f::Symbol)
    if f == :operatordiagonalinverse
        return getoperatordiagonalinverse(jacobi)
    else
        return getfield(jacobi, f)
    end
end

function Base.setproperty!(jacobi::Jacobi, f::Symbol, value)
    if f == :operator
        throw(ReadOnlyMemoryError()) # COV_EXCL_LINE
    else
        return setfield!(jacobi, f, value)
    end
end

# ------------------------------------------------------------------------------
# compute symbols
# ------------------------------------------------------------------------------

"""
```julia
computesymbols(jacobi, ω, θ)
```

Compute or retrieve the symbol matrix for a Jacobi preconditioned operator

# Arguments:

  - `jacobi::Jacobi`:   Jacobi preconditioner to compute symbol matrix for
  - `ω::Array{Real}`:   smoothing weighting factor array
  - `θ::Array{Real}`:   Fourier mode frequency array (one frequency per dimension)

# Returns:

  - symbol matrix for the Jacobi preconditioned operator

# Example:

```jldoctest
using LinearAlgebra

for dimension = 1:3
    # setup
    mesh = []
    if dimension == 1
        mesh = Mesh1D(1.0)
    elseif dimension == 2
        mesh = Mesh2D(1.0, 1.0)
    elseif dimension == 3
        mesh = Mesh3D(1.0, 1.0, 1.0)
    end
    diffusion = GalleryOperator("diffusion", 3, 3, mesh)

    # preconditioner
    jacobi = Jacobi(diffusion)

    # compute symbols
    A = computesymbols(jacobi, [1.0], π * ones(dimension))

    # verify
    using LinearAlgebra
    eigenvalues = real(eigvals(A))
    if dimension == 1
        @assert maximum(eigenvalues) ≈ 1 / 7
    elseif dimension == 2
        @assert minimum(eigenvalues) ≈ -1 / 14
    elseif dimension == 3
        @assert minimum(eigenvalues) ≈ -0.33928571428571486
    end
end

# output

```
"""
function computesymbols(jacobi::Jacobi, ω::Array{<:Real}, θ::Array{<:Real})
    # validate number of parameters
    if length(ω) != 1
        throw(error("exactly one parameter required for Jacobi smoothing")) # COV_EXCL_LINE
    end

    # return
    return I - ω[1] * jacobi.operatordiagonalinverse * computesymbols(jacobi.operator, θ)
end

# ------------------------------------------------------------------------------
