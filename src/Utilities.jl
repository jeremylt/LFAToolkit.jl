# ------------------------------------------------------------------------------
# convenience utilities
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# compute symbol matrix
# ------------------------------------------------------------------------------

"""
```julia
computesymbolsoverrange(
    operator,
    numbersteps1d;
    mass = nothing,
    θ_min = -π / 2,
    θ_band = 2π,
)
```

Compute the eigenvalues and eigenvectors of the symbol matrix for an operator over a range of θ

# Arguments:

  - `operator::Operator`:  finite element operator to compute symbol matrices for
  - `numbersteps1d::Int`:  number of values of θ to sample in each dimension; note: `numbersteps1d`^`dimension` symbol matrices will be computed

# Keyword Arguments:

  - `mass::Union{Operator,Nothing} = nothing`:  mass operator to invert for comparison to analytic solution
  - `θ_min::Real = -π / 2`:                     bottom of range of θ, shifts range to `[θ_min, θ_min + θ_band]`
  - `θ_band::Real = 2π`:                        `θ_max = θ_min + θ_band`

# Returns:

tuple holding

  - values of θ sampled
  - eigenvalues of symbol matrix at θ sampled
  - eigenvectors of symbol matrix at θ sampled

# Examples:

```jldoctest
numbersteps1d = 5;

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

    # compute symbols
    (_, eigenvalues, _) = computesymbolsoverrange(diffusion, numbersteps1d)

    # verify
    eigenvalues = real(eigenvalues)
    if dimension == 1
        @assert minimum(eigenvalues[4, :]) ≈ 1
        @assert maximum(eigenvalues[4, :]) ≈ 4 / 3
    elseif dimension == 2
        @assert minimum(eigenvalues[19, :]) ≈ 2 / 3
        @assert maximum(eigenvalues[19, :]) ≈ 64 / 45
    elseif dimension == 3
        @assert minimum(eigenvalues[94, :]) ≈ 1 / 3
        @assert maximum(eigenvalues[94, :]) ≈ 256 / 225
    end
end

# output

```

```jldoctest
# setup
numbersteps1d = 3;
mesh = Mesh2D(0.5, 0.5);
p = 6;

# operators
mass = GalleryOperator("mass", p + 1, p + 1, mesh);
diffusion = GalleryOperator("diffusion", p + 1, p + 1, mesh);

# compute symbols
(_, eigenvalues, _) =
    computesymbolsoverrange(diffusion, numbersteps1d; mass = mass, θ_min = -π);
eigenvalues = real(eigenvalues);

@assert minimum(eigenvalues[2, :]) ≈ π^2

# output

```
"""
function computesymbolsoverrange(
    operator::Operator,
    numbersteps1d::Int;
    mass::Union{Operator,Nothing} = nothing,
    θ_min::Real = -π / 2,
    θ_band::Real = 2π,
)
    # setup range
    dimension = operator.dimension
    numbersteps = numbersteps1d^dimension
    θ_max = θ_min + θ_band
    θ_range1d = LinRange(θ_min, θ_max, numbersteps1d)
    θ_range = zeros(numbersteps, dimension)

    # setup
    numbermodes = size(operator.rowmodemap)[1]
    eigenvalues = zeros(ComplexF64, numbersteps, numbermodes)
    eigenvectors = zeros(ComplexF64, numbersteps, numbermodes, numbermodes)
    rangeiterator = []
    if dimension == 1
        rangeiterator = θ_range1d
    elseif dimension == 2
        rangeiterator = Iterators.product(θ_range1d, θ_range1d)
    elseif dimension == 3
        rangeiterator = Iterators.product(θ_range1d, θ_range1d, θ_range1d)
    end

    # compute
    for (step, θ) in enumerate(rangeiterator)
        θ = collect(θ)
        A = computesymbols(operator, θ)
        if !isnothing(mass)
            A = computesymbols(mass, θ) \ A
        end
        currenteigen = eigen(A)
        eigenvalues[step, :] = currenteigen.values
        eigenvectors[step, :, :] = currenteigen.vectors
        θ_range[step, :] = θ
    end

    # return
    return (θ_range, eigenvalues, eigenvectors)
end

"""
```julia
computesymbolsoverrange(
    preconditioner,
    ω,
    numbersteps1d;
    mass = nothing,
    θ_min = -π / 2,
    θ_band = 2π,
)
```

Compute the eigenvalues and eigenvectors of the symbol matrix for a preconditioned operator over a range of θ

# Arguments:

  - `preconditioner::AbstractPreconditioner`:  preconditioner to compute symbol matries for
  - `ω::Array`:                                smoothing parameter array
  - `numbersteps1d::Int`:                      number of values of θ to sample in each dimension; note: `numbersteps1d`^`dimension` symbol matrices will be computed

# Keyword Arguments:

  - `mass::Union{Operator,Nothing} = nothing`:  mass operator to invert for comparison to analytic solution
  - `θ_min::Real = -π / 2`:                     bottom of range of θ, shifts range to `[θ_min, θ_min + θ_band]`
  - `θ_band::Real = 2π`:                        `θ_max = θ_min + θ_band`

# Returns:

tuple holding

  - values of θ sampled
  - eigenvalues of symbol matrix at θ sampled
  - eigenvectors of symbol matrix at θ sampled

# Example:

```jldoctest
numbersteps1d = 5;

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
    (_, eigenvalues, _) = computesymbolsoverrange(chebyshev, [1], numbersteps1d)

    # verify
    eigenvalues = real(eigenvalues)
    if dimension == 1
        @assert minimum(eigenvalues[4, :]) ≈ 0.15151515151515105
        @assert maximum(eigenvalues[4, :]) ≈ 0.27272727272727226
    elseif dimension == 2
        @assert minimum(eigenvalues[19, :]) ≈ -0.25495098334134725
        @assert maximum(eigenvalues[19, :]) ≈ -0.17128758445192374
    elseif dimension == 3
        @assert minimum(eigenvalues[94, :]) ≈ -0.8181818181818181
        @assert maximum(eigenvalues[94, :]) ≈ -0.357575757575757
    end
end

# output

```
"""
function computesymbolsoverrange(
    preconditioner::AbstractPreconditioner,
    ω::Array,
    numbersteps1d::Int;
    mass::Union{Operator,Nothing} = nothing,
    θ_min::Real = -π / 2,
    θ_band::Real = 2π,
)
    # setup range
    dimension = preconditioner.operator.dimension
    numbersteps = numbersteps1d^dimension
    θ_max = θ_min + θ_band
    θ_range1d = LinRange(θ_min, θ_max, numbersteps1d)
    θ_range = zeros(numbersteps, dimension)

    # setup
    numbermodes = size(preconditioner.operator.rowmodemap)[1]
    eigenvalues = zeros(ComplexF64, numbersteps, numbermodes)
    eigenvectors = zeros(ComplexF64, numbersteps, numbermodes, numbermodes)
    rangeiterator = []
    if dimension == 1
        rangeiterator = θ_range1d
    elseif dimension == 2
        rangeiterator = Iterators.product(θ_range1d, θ_range1d)
    elseif dimension == 3
        rangeiterator = Iterators.product(θ_range1d, θ_range1d, θ_range1d)
    end

    # compute
    for (step, θ) in enumerate(rangeiterator)
        θ = collect(θ)
        A = computesymbols(preconditioner, ω, θ)
        if !isnothing(mass)
            A = computesymbols(mass, θ) \ A
        end
        currenteigen = eigen(A)
        eigenvalues[step, :] = currenteigen.values
        eigenvectors[step, :, :] = currenteigen.vectors
        θ_range[step, :] = θ
    end

    # return
    return (θ_range, eigenvalues, eigenvectors)
end

"""
```julia
computesymbolsoverrange(
    multigrid,
    p,
    v,
    numbersteps1d;
    mass = nothing,
    θ_min = -π / 2,
    θ_band = 2π,
)
```

Compute the eigenvalues and eigenvectors of the symbol matrix for a multigrid preconditioned operator over a range of θ

# Arguments:

  - `multigrid::Multigrid`:  preconditioner to compute symbol matries for
  - `p::Array{Real}`:        smoothing parameter array
  - `v::Array{Int}`:         pre and post smooths iteration count array, 0 indicates no pre or post smoothing
  - `numbersteps1d::Int`:    number of values of θ to sample in each dimension; note: `numbersteps1d`^`dimension` symbol matrices will be computed

# Keyword Arguments:

  - `mass::Union{Operator,Nothing} = nothing`:  mass operator to invert for comparison to analytic solution
  - `θ_min::Real = -π / 2`:                     bottom of range of θ, shifts range to `[θ_min, θ_min + θ_band]`
  - `θ_band::Real = 2π`:                        `θ_max = θ_min + θ_band`

# Returns:

tuple holding

  - values of θ sampled
  - eigenvalues of symbol matrix at θ sampled
  - eigenvectors of symbol matrix at θ sampled

# Example:

```jldoctest
numbersteps1d = 5;

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
    ctofbasis = TensorH1LagrangeBasis(3, 5, 1, dimension; collocatedquadrature = true)

    # operators
    finediffusion = GalleryOperator("diffusion", 5, 5, mesh)
    coarsediffusion = GalleryOperator("diffusion", 3, 5, mesh)

    # smoother
    jacobi = Jacobi(finediffusion)

    # preconditioner
    multigrid = PMultigrid(finediffusion, coarsediffusion, jacobi, [ctofbasis])

    # compute symbols
    (_, eigenvalues, _) = computesymbolsoverrange(multigrid, [1.0], [1, 1], numbersteps1d)

    # verify
    eigenvalues = real(eigenvalues)
    if dimension == 1
        @assert maximum(eigenvalues[4, :]) ≈ 0.64
    elseif dimension == 2
        @assert maximum(eigenvalues[19, :]) ≈ 0.9082562365654528
    elseif dimension == 3
        @assert maximum(eigenvalues[94, :]) ≈ 1.4359882222222669
    end
end

# output

```
"""
function computesymbolsoverrange(
    multigrid::Multigrid,
    p::Array{<:Real},
    v::Array{Int},
    numbersteps1d::Int;
    mass::Union{Operator,Nothing} = nothing,
    θ_min::Real = -π / 2,
    θ_band::Real = 2π,
)
    # setup range
    dimension = multigrid.fineoperator.dimension
    numbersteps = numbersteps1d^dimension
    θ_max = θ_min + θ_band
    θ_range1d = LinRange(θ_min, θ_max, numbersteps1d)
    θ_range = zeros(numbersteps, dimension)

    # setup
    numbermodes = size(multigrid.fineoperator.rowmodemap)[1]
    eigenvalues = zeros(ComplexF64, numbersteps, numbermodes)
    eigenvectors = zeros(ComplexF64, numbersteps, numbermodes, numbermodes)
    rangeiterator = []
    if dimension == 1
        rangeiterator = θ_range1d
    elseif dimension == 2
        rangeiterator = Iterators.product(θ_range1d, θ_range1d)
    elseif dimension == 3
        rangeiterator = Iterators.product(θ_range1d, θ_range1d, θ_range1d)
    end

    # compute
    for (step, θ) in enumerate(rangeiterator)
        θ = [abs(θ_i) > 1000 * eps() ? θ_i : 1000 * eps() for θ_i in θ]
        A = computesymbols(multigrid, p, v, θ)
        if !isnothing(mass)
            A = computesymbols(mass, θ) \ A
        end
        currenteigen = eigen(A)
        eigenvalues[step, :] = currenteigen.values
        eigenvectors[step, :, :] = currenteigen.vectors
        θ_range[step, :] = θ
    end

    # return
    return (θ_range, eigenvalues, eigenvectors)
end

# ------------------------------------------------------------------------------
