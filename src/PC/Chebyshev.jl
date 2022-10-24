# ------------------------------------------------------------------------------
# Chebyshev preconditioner
# ------------------------------------------------------------------------------

"""
```julia
Chebyshev(operator)
```

Chebyshev polynomial preconditioner for finite element operators.
The Chebyshev semi-iterative method is applied to the matrix ``D^{-1} A``, where ``D^{-1}`` is the inverse of the operator diagonal.

# Arguments:

  - `operator::Operator`:  finite element operator to precondition

# Returns:

  - Chebyshev preconditioner object

# Example:

```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
mass = GalleryOperator("mass", 4, 4, mesh);

# preconditioner
chebyshev = Chebyshev(mass);

# verify
println(chebyshev)

# output

chebyshev preconditioner:
eigenvalue estimates:
  estimated minimum 0.2500
  estimated maximum 1.3611
estimate scaling:
  λ_min = a * estimated min + b * estimated max
  λ_max = c * estimated min + d * estimated max
  a = 0.0000
  b = 0.1000
  c = 0.0000
  d = 1.0000
```
"""
mutable struct Chebyshev <: AbstractPreconditioner
    # data never changed
    operator::Operator
    eigenvaluebounds::Array{Float64,1}

    # data empty until assembled
    eigenvalueestimates::Array{Float64,1}
    operatordiagonalinverse::AbstractArray{Float64}

    # inner constructor
    Chebyshev(operator::Operator) = new(operator, [0.0, 0.1, 0.0, 1.0])
end

# printing
# COV_EXCL_START
function Base.show(io::IO, chebyshev::Chebyshev)
    print(io, "chebyshev preconditioner:")

    # eigenvalue estimates
    print(io, "\neigenvalue estimates:")
    @printf(io, "\n  estimated minimum %.4f", chebyshev.eigenvalueestimates[1])
    @printf(io, "\n  estimated maximum %.4f", chebyshev.eigenvalueestimates[2])

    # estimate scaling
    print(io, "\nestimate scaling:")
    print(io, "\n  λ_min = a * estimated min + b * estimated max")
    print(io, "\n  λ_max = c * estimated min + d * estimated max")
    @printf(io, "\n  a = %.4f", chebyshev.eigenvaluebounds[1])
    @printf(io, "\n  b = %.4f", chebyshev.eigenvaluebounds[2])
    @printf(io, "\n  c = %.4f", chebyshev.eigenvaluebounds[3])
    @printf(io, "\n  d = %.4f", chebyshev.eigenvaluebounds[4])
end
# COV_EXCL_STOP

# ------------------------------------------------------------------------------
# data for computing symbols
# ------------------------------------------------------------------------------

"""
```julia
getoperatordiagonalinverse(chebyshev)
```

Compute or retrieve the inverse of the symbol matrix diagonal for a Chebyshev preconditioner

# Returns:

  - symbol matrix diagonal inverse for the operator

# Example:

```jldoctest
# setup
mesh = Mesh1D(1.0);
diffusion = GalleryOperator("diffusion", 3, 3, mesh);

# preconditioner
chebyshev = Chebyshev(diffusion);

# verify operator diagonal inverse
@assert chebyshev.operatordiagonalinverse ≈ [6/7 0; 0 3/4]

# output

```
"""
function getoperatordiagonalinverse(chebyshev::Chebyshev)
    # assemble if needed
    if !isdefined(chebyshev, :operatordiagonalinverse)
        # retrieve diagonal and invert
        diagonalinverse = chebyshev.operator.diagonal^-1

        # store
        chebyshev.operatordiagonalinverse = diagonalinverse
    end

    # return
    return getfield(chebyshev, :operatordiagonalinverse)
end

"""
```julia
geteigenvalueestimates(chebyshev)
```

Compute or retrieve the eigenvalue estimates for a Chebyshev preconditioner

# Returns:

  - eigenvalue estimates for the operator

# Example:

```jldoctest
# setup
mesh = Mesh1D(1.0);
diffusion = GalleryOperator("diffusion", 3, 3, mesh);

# preconditioner
chebyshev = Chebyshev(diffusion);

# verify eigenvalue estimates
@assert chebyshev.eigenvalueestimates ≈ [0, 15 / 7]

# output

```
"""
function geteigenvalueestimates(chebyshev::Chebyshev)
    # assemble if needed
    if !isdefined(chebyshev, :eigenvalueestimates)
        dimension = chebyshev.operator.dimension
        λ_min = 1
        λ_max = 0
        θ_step = 2π / 4
        θ_range = -π/2:θ_step:3π/2-θ_step

        # compute eigenvalues
        if dimension == 1
            for θ_x in θ_range
                A = computesymbols(chebyshev.operator, [θ_x])
                eigenvalues = abs.(eigvals(chebyshev.operatordiagonalinverse * A),)
                λ_min = minimum([λ_min, eigenvalues...])
                λ_max = maximum([λ_max, eigenvalues...])
            end
        elseif dimension == 2
            for θ_x in θ_range, θ_y in θ_range
                A = computesymbols(chebyshev.operator, [θ_x, θ_y])
                eigenvalues = abs.(eigvals(chebyshev.operatordiagonalinverse * A),)
                λ_min = minimum([λ_min, eigenvalues...])
                λ_max = maximum([λ_max, eigenvalues...])
            end
        elseif dimension == 3
            for θ_x in θ_range, θ_y in θ_range, θ_z in θ_range
                A = computesymbols(chebyshev.operator, [θ_x, θ_y, θ_z])
                eigenvalues = abs.(eigvals(chebyshev.operatordiagonalinverse * A),)
                λ_min = minimum([λ_min, eigenvalues...])
                λ_max = maximum([λ_max, eigenvalues...])
            end
        end

        # store
        chebyshev.eigenvalueestimates = [λ_min, λ_max]
    end

    # return
    return getfield(chebyshev, :eigenvalueestimates)
end

"""
```julia
seteigenvalueestimatescaling(chebyshev, eigenvaluebounds)
```

Set the scaling of the eigenvalue estimates for a Chebyshev preconditioner

# Arguments:

  - `chebyshev::Chebyshev`:                preconditioner to set eigenvalue estimate scaling
  - `eigenvaluebounds::Array{Float64,1}`:  array of 4 scaling factors to use when setting ``\\lambda_{\\text{min}}`` and ``\\lambda_{\\text{max}}`` based on eigenvalue estimates

``\\lambda_{\\text{min}}`` = a * estimated min + b * estimated max

``\\lambda_{\\text{max}}`` = c * estimated min + d * estimated max

# Example:

```jldoctest
# setup
mesh = Mesh1D(1.0);
diffusion = GalleryOperator("diffusion", 3, 3, mesh);

# preconditioner
chebyshev = Chebyshev(diffusion)

# set eigenvalue estimate scaling
# PETSc default is to use 1.1, 0.1 of max eigenvalue estimate
#   https://www.mcs.anl.gov/petsc/petsc-3.3/docs/manualpages/KSP/KSPChebyshevSetEstimateEigenvalues.html
seteigenvalueestimatescaling(chebyshev, [0.0, 0.1, 0.0, 1.1]);
println(chebyshev)

# output

chebyshev preconditioner:
eigenvalue estimates:
  estimated minimum 0.0000
  estimated maximum 2.1429
estimate scaling:
  λ_min = a * estimated min + b * estimated max
  λ_max = c * estimated min + d * estimated max
  a = 0.0000
  b = 0.1000
  c = 0.0000
  d = 1.1000
```
"""
function seteigenvalueestimatescaling(
    chebyshev::Chebyshev,
    eigenvaluebounds::Array{Float64,1},
)
    if length(eigenvaluebounds) != 4
        throw(error("exactly four transformation arguments are required")) # COV_EXCL_LINE
    end

    chebyshev.eigenvaluebounds[1] = eigenvaluebounds[1]
    chebyshev.eigenvaluebounds[2] = eigenvaluebounds[2]
    chebyshev.eigenvaluebounds[3] = eigenvaluebounds[3]
    chebyshev.eigenvaluebounds[4] = eigenvaluebounds[4]
    chebyshev
end

# ------------------------------------------------------------------------------
# get/set property
# ------------------------------------------------------------------------------

function Base.getproperty(preconditioner::Chebyshev, f::Symbol)
    if f == :operatordiagonalinverse
        return getoperatordiagonalinverse(preconditioner)
    elseif f == :eigenvalueestimates
        return geteigenvalueestimates(preconditioner)
    else
        return getfield(preconditioner, f)
    end
end

function Base.setproperty!(preconditioner::Chebyshev, f::Symbol, value)
    if f == :operator
        throw(ReadOnlyMemoryError()) # COV_EXCL_LINE
    elseif f == :eigenvaluebounds
        throw(ReadOnlyMemoryError()) # COV_EXCL_LINE
    else
        return setfield!(preconditioner, f, value)
    end
end

# ------------------------------------------------------------------------------
# compute symbols
# ------------------------------------------------------------------------------

"""
```julia
computesymbols(chebyshev, ω, θ)
```

Compute or retrieve the symbol matrix for a Chebyshev preconditioned operator

# Arguments:

  - `chebyshev::Chebyshev`:  Chebyshev preconditioner to compute symbol matrix for
  - `ω::Array{Real}`:        smoothing parameter array [degree], [degree, ``\\lambda_{\\text{max}}``], or [degree, ``\\lambda_{\\text{min}}``, ``\\lambda_{\\text{max}}``]
  - `θ::Array{Real}`:        Fourier mode frequency array (one frequency per dimension)

# Returns:

  - symbol matrix for the Chebyshev preconditioned operator

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
    chebyshev = Chebyshev(diffusion)

    # compute symbols
    A = computesymbols(chebyshev, [1], π * ones(dimension))

    # verify
    using LinearAlgebra
    eigenvalues = real(eigvals(A))
    if dimension == 1
        @assert minimum(eigenvalues) ≈ 0.15151515151515105
        @assert maximum(eigenvalues) ≈ 0.27272727272727226
    elseif dimension == 2
        @assert minimum(eigenvalues) ≈ -0.25495098334134725
        @assert maximum(eigenvalues) ≈ -0.17128758445192374
    elseif dimension == 3
        @assert minimum(eigenvalues) ≈ -0.8181818181818181
        @assert maximum(eigenvalues) ≈ -0.357575757575757
    end
end

# output

```
"""
function computesymbols(chebyshev::Chebyshev, ω::Array{<:Real}, θ::Array{<:Real})
    # validate number of parameters
    if length(ω) < 1
        throw(error("at least one parameter required for Chebyshev smoothing")) # COV_EXCL_LINE
    elseif length(ω) > 3
        throw(error("no more than three parameters allowed for Chebyshev smoothing")) # COV_EXCL_LINE
    end
    if (ω[1] % 1) > 1E-14 || ω[1] < 1
        throw(error("first parameter must be degree of Chebyshev smoother")) # COV_EXCL_LINE
    end

    # get operator symbol
    A = computesymbols(chebyshev.operator, θ)

    # set eigenvalue estimates
    λ_min = 0
    λ_max = 0
    if length(ω) == 1
        λ_min = chebyshev.eigenvalueestimates[1]
        λ_max = chebyshev.eigenvalueestimates[2]
    elseif length(ω) == 2
        λ_max = ω[2]
    else
        λ_min = ω[2]
        λ_max = ω[3]
    end
    lower = λ_min * chebyshev.eigenvaluebounds[1] + λ_max * chebyshev.eigenvaluebounds[2]
    upper = λ_min * chebyshev.eigenvaluebounds[3] + λ_max * chebyshev.eigenvaluebounds[4]

    # compute Chebyshev smoother of given degree
    D_inv = chebyshev.operatordiagonalinverse
    D_inv_A = D_inv * A
    k = ω[1] # degree of Chebyshev smoother
    α = (upper + lower) / 2
    c = (upper - lower) / 2
    β_0 = -c^2 / (2 * α)
    γ_1 = -(α + β_0)
    E_0 = I
    E_1 = I - (1 / α) * D_inv_A * E_0
    E_n = I
    for _ = 2:k
        E_n = (D_inv_A * E_1 - α * E_1 - β_0 * E_0) / γ_1
        β_0 = (c / 2)^2 / γ_1
        γ_1 = -(α + β_0)
        E_0 = E_1
        E_1 = E_n
    end

    # return
    return E_1
end

# ------------------------------------------------------------------------------
