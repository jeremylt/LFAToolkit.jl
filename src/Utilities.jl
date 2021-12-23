# ------------------------------------------------------------------------------
# convenience utilities
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# compute symbol matrix
# ------------------------------------------------------------------------------

"""
```julia
computesymbols(operator, θ)
```

Compute the eigenvalues and eigenvectors of the symbol matrix for an operator over
  a range of θ from -π/2 to 3π/2

# Arguments:
- `operator`:      Finite element operator to compute symbol matrices for
- `numbersteps1d`: Number of values of θ to sample in each dimension
                     Note: numbersteps1d^dimension symbol matrices will be computed

# Returns:
- Values of θ sampled
- Eigenvalues of symbol matrix at θ sampled
- Eigenvectors of symbol matrix at θ sampled

# Example:
```jldoctest
using LFAToolkit;

numbersteps1d = 5

for dimension in 1:3
    # setup
    mesh = []
    if dimension == 1
        mesh = Mesh1D(1.0);
    elseif dimension == 2
        mesh = Mesh2D(1.0, 1.0);
    elseif dimension == 3
        mesh = Mesh3D(1.0, 1.0, 1.0);
    end
    diffusion = GalleryOperator("diffusion", 3, 3, mesh);

    # compute symbols
    (_, eigenvalues, _) = computesymbolsoverrange(diffusion, numbersteps1d);

    # verify
    eigenvalues = real(eigenvalues);
    if dimension == 1
        @assert min(eigenvalues[4, :]...) ≈ 1
        @assert max(eigenvalues[4, :]...) ≈ 4/3
    elseif dimension == 2
        @assert min(eigenvalues[19, :]...) ≈ 2/3
        @assert max(eigenvalues[19, :]...) ≈ 64/45
    elseif dimension == 3
        @assert min(eigenvalues[94, :]...) ≈ 1/3
        @assert max(eigenvalues[94, :]...) ≈ 256/225
    end
end

# output

```
"""
function computesymbolsoverrange(operator::Operator, numbersteps1d::Int)
    # setup range
    dimension = operator.dimension
    numbersteps = numbersteps1d^dimension
    θ_min = -π / 2
    θ_max = 3π / 2
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
computesymbols(operator, θ)
```

Compute the eigenvalues and eigenvectors of the symbol matrix for an operator
  over a range of θ from -π/2 to 3π/2

# Arguments:
- `preconditioner`: Preconditioner to compute symbol matries for
- `ω`:              Smoothing parameter array
- `numbersteps1d`:  Number of values of θ to sample in each dimension
                      Note: numbersteps1d^dimension symbol matrices will be computed

# Returns:
- Values of θ sampled
- Eigenvalues of symbol matrix at θ sampled
- Eigenvectors of symbol matrix at θ sampled

# Example:
```jldoctest
using LFAToolkit;

numbersteps1d = 5

for dimension in 1:3
    # setup
    mesh = []
    if dimension == 1
        mesh = Mesh1D(1.0);
    elseif dimension == 2
        mesh = Mesh2D(1.0, 1.0);
    elseif dimension == 3
        mesh = Mesh3D(1.0, 1.0, 1.0);
    end
    diffusion = GalleryOperator("diffusion", 3, 3, mesh);

    # preconditioner
    chebyshev = Chebyshev(diffusion);

    # compute symbols
    (_, eigenvalues, _) = computesymbolsoverrange(chebyshev, [1], numbersteps1d);

    # verify
    eigenvalues = real(eigenvalues);
    if dimension == 1
        @assert min(eigenvalues[4, :]...) ≈ 0.15151515151515105
        @assert max(eigenvalues[4, :]...) ≈ 0.27272727272727226
    elseif dimension == 2
        @assert min(eigenvalues[19, :]...) ≈ -0.25495098334134725
        @assert max(eigenvalues[19, :]...) ≈ -0.17128758445192374
    elseif dimension == 3
        @assert min(eigenvalues[94, :]...) ≈ -0.8181818181818181
        @assert max(eigenvalues[94, :]...) ≈ -0.357575757575757
    end
end

# output

```
"""
function computesymbolsoverrange(
    preconditioner::AbstractPreconditioner,
    ω::Array,
    numbersteps1d::Int,
)
    # setup range
    dimension = preconditioner.operator.dimension
    numbersteps = numbersteps1d^dimension
    θ_min = -π / 2
    θ_max = 3π / 2
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
        currenteigen = eigen(A)
        eigenvalues[step, :] = currenteigen.values
        eigenvectors[step, :, :] = currenteigen.vectors
        θ_range[step, :] = θ
    end

    # return
    return (θ_range, eigenvalues, eigenvectors)
end

# ------------------------------------------------------------------------------
