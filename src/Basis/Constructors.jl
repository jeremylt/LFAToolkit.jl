# ------------------------------------------------------------------------------
# finite element bases
# ------------------------------------------------------------------------------

using Polynomials

# ------------------------------------------------------------------------------
# conformal map functions for generating transformed polynomial bases
# ------------------------------------------------------------------------------

"""
```julia
sausage(d)
```
# Arguments:
- `d`: polynomial degree of truncated Taylor series expansion of arcsin(s).

# Returns:
Conformal mapping of Gauss ellipses to sausages using a truncated Taylor expansion of arcsin(s)
See Figure 4.1 of Hale and Trefethen (2008).

# Example:
```jldoctest
# sausage conformal map
g, derivative(g) = LFAToolkit.sausage(9);
```
"""
function sausage(d)
    c = zeros(d + 1)
    c[2:2:end] = [1, cumprod(1:2:d-2) ./ cumprod(2:2:d-1)...] ./ (1:2:d)
    c /= sum(c)
    g = Polynomial(c)
    g_prime = derivative(g)
    g, g_prime
end

"""
```julia
kosloff_tal_ezer(α)
```
# Arguments:
- `α`: polynomial degree of truncated Taylor series expansion of arcsin(s).

# Returns:
The Kosloff and Tal-Ezer conformal map derived from the inverse sine function.

# Example:
```jldoctest
# kosloff_tal_ezer conformal map
g, g_prime = LFAToolkit.kosloff_tal_ezer(0.95);
``` 
"""
function kosloff_tal_ezer(α)
    g(s) = asin(α * s) / asin(α)
    g_prime(s) = α / (asin(α) * sqrt(1 - (α * s)^2))
    g, g_prime
end

"""
```julia
hale_trefethen_strip(ρ)
```
# Arguments:
- `ρ`: sum of the semiminor and semimajor axis  

# Returns:
The Hale and Trefethen strip transformation

# Example:
```jldoctest
# hale_trefethen_strip conformal map
g, g_prime = LFAToolkit.hale_trefethen_strip(1.4);
```  
"""
function hale_trefethen_strip(ρ)
    τ = π / log(ρ)
    d = 0.5 + 1 / (exp(τ * π) + 1)
    π2 = π / 2
    # Unscaled functions of u
    g_u(u) = log(1 + exp(-τ * (π2 + u))) - log(1 + exp(-τ * (π2 - u))) + d * τ * u
    g_prime_u(u) = 1 / (exp(τ * (π2 + u)) + 1) + 1 / (exp(τ * (π2 - u)) + 1) - d
    # Normalizing factor and scaled functions of s
    C = 1 / g_u(π / 2)
    g(s) = C * g_u(asin(s))
    g_prime(s) = -τ * C / sqrt(1 - s^2) * g_prime_u(asin(s))
    g, g_prime
end

"""
```julia
transformquadrature(points, weights, mapping)
```
# Arguments:
- `points`:  number quadrature points
- `weights`: boolean flag indicating if quadrature weights are desired
- `mapping`: choice of conformal map

# Returns:
Transformed quadrature by applying a smooth mapping = (g, g_prime) from the original domain.

# Example:
```jldoctest
# generate transformed quadrature points, weights with choice of conformal map
m_points, m_weights = LFAToolkit.transformquadrature(3, true, mapping);
```
"""
function transformquadrature(points, weights = nothing, mapping = nothing)
    if isnothing(mapping)
        if isnothing(weights)
            return points
        else
            return points, weights
        end
    end
    g_map, g_map_prime = mapping
    m_points = g_map.(points)
    if !isnothing(weights)
        m_weights = g_map_prime.(points) .* weights
        return m_points, m_weights
    else
        return m_points
    end
end

# ------------------------------------------------------------------------------
# utility functions for generating polynomial bases
# ------------------------------------------------------------------------------

"""
```julia
gaussquadrature(q, mapping)
```

Construct a Gauss-Legendre quadrature with the option of applying Conformal maps

# Arguments:
- `q`:  number of Gauss-Legendre points
- `mapping`: choice of conformal map

# Returns:
- Gauss-Legendre quadrature points, weights and mapped version

# Example:
```jldoctest
# generate Gauss-Legendre points and weights
quadraturepoints, quadratureweights = LFAToolkit.gaussquadrature(5);

# generate Gauss-Legendre points, weights and mapped version
quadraturepoints, quadratureweights = LFAToolkit.gaussquadrature(5, mapping);

# verify
truepoints = [
    -√(5 + 2*√(10/7))/3,
    -√(5 - 2*√(10/7))/3,
    0.0,
    √(5 - 2*√(10/7))/3,
    √(5 + 2*√(10/7))/3
];
@assert truepoints ≈ quadraturepoints

trueweights = [
    (322-13*√70)/900,
    (322+13*√70)/900,
    128/225,
    (322+13*√70)/900,
    (322-13*√70)/900
];
@assert trueweights ≈ quadratureweights

# output

```
"""
function gaussquadrature(q::Int; mapping = nothing)
    quadraturepoints = zeros(Float64, q)
    quadratureweights = zeros(Float64, q)

    if q < 1
        throw(DomainError(q, "q must be greater than or equal to 1")) # COV_EXCL_LINE
    end

    # build qref1d, qweight1d
    for i = 0:floor(Int, q / 2)
        # guess
        xi = cos(π * (2 * i + 1.0) / (2 * q))

        # Pn(xi)
        p0 = 1.0
        p1 = xi
        p2 = 0.0
        for j = 2:q
            p2 = ((2 * j - 1.0) * xi * p1 - (j - 1.0) * p0) / j
            p0 = p1
            p1 = p2
        end

        # first Newton step
        dp2 = (xi * p2 - p0) * q / (xi * xi - 1.0)
        xi = xi - p2 / dp2

        # Newton to convergence
        iter = 0
        maxiter = q^2 * 100
        while iter < maxiter && abs(p2) > 1e-15
            p0 = 1.0
            p1 = xi
            for j = 2:q
                p2 = ((2 * j - 1.0) * xi * p1 - (j - 1.0) * p0) / j
                p0 = p1
                p1 = p2
            end
            dp2 = (xi * p2 - p0) * q / (xi * xi - 1.0)
            xi = xi - p2 / dp2
            iter += 1
        end

        # save xi, wi
        quadraturepoints[i+1] = -xi
        quadraturepoints[q-i] = xi
        wi = 2.0 / ((1.0 - xi * xi) * dp2 * dp2)
        quadratureweights[i+1] = wi
        quadratureweights[q-i] = wi
    end
    quadraturepoints[abs.(quadraturepoints).<10*eps()] .= 0

    return transformquadrature(quadraturepoints, quadratureweights, mapping)
end

"""
```julia
gausslobattoquadrature(q, weights, mapping)
```

Construct a Gauss-Legendre-Lobatto quadrature with the option of applying Conformal maps

# Arguments:
- `q`:        number of Gauss-Legendre-Lobatto points
- `weights`:  boolean flag indicating if quadrature weights are desired
- `mapping`: choice of conformal map

# Returns:
- Gauss-Legendre-Lobatto quadrature points or points, weights and mapped version

# Example:
```jldoctest
# generate Gauss-Legendre-Lobatto points
quadraturepoints = LFAToolkit.gausslobattoquadrature(5, false);

# verify
truepoints = [-1.0, -√(3/7), 0.0, √(3/7), 1.0];
@assert truepoints ≈ quadraturepoints

# generate Gauss-Legendre-Lobatto points and weights
quadraturepoints, quadratureweights = LFAToolkit.gausslobattoquadrature(5, true);

# verify
trueweights = [1/10, 49/90, 32/45, 49/90, 1/10];
@assert trueweights ≈ quadratureweights

# output

```
"""
function gausslobattoquadrature(q::Int, weights::Bool; mapping = nothing)
    quadraturepoints = zeros(Float64, q)
    quadratureweights = zeros(Float64, q)

    if q < 2
        throw(DomainError(q, "q must be greater than or equal to 2")) # COV_EXCL_LINE
    end

    # endpoints
    quadraturepoints[1] = -1.0
    quadraturepoints[q] = 1.0
    if weights
        wi = 2.0 / (q * (q - 1.0))
        quadratureweights[1] = wi
        quadratureweights[q] = wi
    end

    # build qref1d, qweight1d
    for i = 1:floor(Int, (q - 1) / 2)
        # guess
        xi = cos(π * i / (q - 1.0))

        # Pn(xi)
        p0 = 1.0
        p1 = xi
        p2 = 0.0
        for j = 2:q-1
            p2 = ((2 * j - 1.0) * xi * p1 - (j - 1.0) * p0) / j
            p0 = p1
            p1 = p2
        end

        # first Newton step
        dp2 = (xi * p2 - p0) * q / (xi * xi - 1.0)
        d2p2 = (2 * xi * dp2 - q * (q - 1.0) * p2) / (1.0 - xi * xi)
        xi = xi - dp2 / d2p2

        # Newton to convergence
        iter = 0
        maxiter = q^2 * 100
        while iter < maxiter && abs(dp2) > 1e-15
            p0 = 1.0
            p1 = xi
            for j = 2:q-1
                p2 = ((2 * j - 1.0) * xi * p1 - (j - 1.0) * p0) / j
                p0 = p1
                p1 = p2
            end
            dp2 = (xi * p2 - p0) * q / (xi * xi - 1.0)
            d2p2 = (2 * xi * dp2 - q * (q - 1.0) * p2) / (1.0 - xi * xi)
            xi = xi - dp2 / d2p2
            iter += 1
        end

        # save xi, wi
        quadraturepoints[i+1] = -xi
        quadraturepoints[q-i] = xi
        if weights
            wi = 2.0 / (q * (q - 1.0) * p2 * p2)
            quadratureweights[i+1] = wi
            quadratureweights[q-i] = wi
        end
    end
    quadraturepoints[abs.(quadraturepoints).<10*eps()] .= 0


    # return
    if weights
        return transformquadrature(quadraturepoints, quadratureweights, mapping)
    else
        return transformquadrature(quadraturepoints, mapping)
    end
end

"""
```julia
buildinterpolationandgradient(
    nodes,
    quadraturepoints,
)
```

Build one dimensional interpolation and gradient matrices, from Fornberg 1998

# Arguments:
- `nodes`:             1d basis nodes
- `quadraturepoints`:  1d basis quadrature points

# Returns:
- One dimensional interpolation and gradient matrices

# Example:
```jldoctest
# get nodes, quadrature points, and weights
numbernodes = 3;
numberquadraturepoints = 4;
nodes = LFAToolkit.gausslobattoquadrature(numbernodes, false);
quadraturepoints, quadratureweights1d = LFAToolkit.gaussquadrature(numberquadraturepoints);

# build interpolation, gradient matrices
interpolation, gradient = LFAToolkit.buildinterpolationandgradient(nodes, quadraturepoints);

# verify
for i in 1:numberquadraturepoints
    total = sum(interpolation[i, :]);
    @assert total ≈ 1.0

    total = sum(gradient[i, :]);
    @assert abs(total) < 1e-14
end

# output

```
"""
function buildinterpolationandgradient(
    nodes1d::AbstractArray{Float64},
    quadraturepoints1d::AbstractArray{Float64},
)
    # check inputs
    numbernodes1d = length(nodes1d)
    numberquadratuepoints1d = length(quadraturepoints1d)
    if numbernodes1d < 2
        # COV_EXCL_START
        throw(
            DomanError(
                numbernodes1d,
                "length of nodes1d must be greater than or equal to 2",
            ),
        )
        # COV_EXCL_STOP
    end
    if numbernodes1d < 2
        # COV_EXCL_START
        throw(
            DomanError(
                numbernodes1d,
                "length of quadraturepoints1d must be greater than or equal to 2",
            ),
        )
        # COV_EXCL_STOP
    end

    # build interpolation, gradient matrices
    # Fornberg, 1998
    interpolation1d = zeros(Float64, numberquadratuepoints1d, numbernodes1d)
    gradient1d = zeros(Float64, numberquadratuepoints1d, numbernodes1d)
    for i = 1:numberquadratuepoints1d
        c1 = 1.0
        c3 = nodes1d[1] - quadraturepoints1d[i]
        interpolation1d[i, 1] = 1.0
        for j = 2:numbernodes1d
            c2 = 1.0
            c4 = c3
            c3 = nodes1d[j] - quadraturepoints1d[i]
            for k = 1:j-1
                dx = nodes1d[j] - nodes1d[k]
                c2 *= dx
                if k == j - 1
                    gradient1d[i, j] =
                        c1 * (interpolation1d[i, k] - c4 * gradient1d[i, k]) / c2
                    interpolation1d[i, j] = -c1 * c4 * interpolation1d[i, k] / c2
                end
                gradient1d[i, k] = (c3 * gradient1d[i, k] - interpolation1d[i, k]) / dx
                interpolation1d[i, k] = c3 * interpolation1d[i, k] / dx
            end
            c1 = c2
        end
    end

    # return
    return interpolation1d, gradient1d
end

# ------------------------------------------------------------------------------
# user utility constructors
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# -- single element bases
# ------------------------------------------------------------------------------

"""
```julia
TensorH1LagrangeBasis(
    numbernodes1d,
    numberquadraturepoints1d,
    numbercomponents,
    dimension;
    collocatedquadrature = false,
    mapping = nothing,
)
```

Tensor product basis on Gauss-Legendre-Lobatto points with Gauss-Legendre (default)
  or Gauss-Legendre-Lobatto quadrature points

# Arguments:
- `numbernodes1d`:             number of Gauss-Legendre-Lobatto nodes in 1 dimension
- `numberquadraturepoints1d`:  number of quadrature points in 1 dimension
- `numbercomponents`:          number of components
- `dimension`:                 dimension of basis

# Keyword Arguments:
- `collocatedquadrature = false`:  Gauss-Legendre or Gauss-Legendre-Lobatto quadrature points,
                                       default: false, Gauss-Legendre-Lobatto

# Keyword Arguments:                                       
- `mapping = nothing`:  sausage, kosloff_tal_ezer, hale_trefethen_strip

# Returns:
- H1 Lagrange tensor product basis object

# Example:
```jldoctest
# generate H1 Lagrange tensor product basis
basis = TensorH1LagrangeBasis(4, 4, 3, 2);

# generate basis with Gauss-Legendre quadrature points
basis = TensorH1LagrangeBasis(4, 4, 3, 2; collocatedquadrature=true);

# generate transformed basis from conformal maps with Gauss-Legendre quadrature points
basis = TensorH1LagrangeBasis(4, 4, 3, 2, collocatedquadrature=true, mapping);

# verify
println(basis)

# output
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 3
    dimension: 2
```
"""
function TensorH1LagrangeBasis(
    numbernodes1d::Int,
    numberquadraturepoints1d::Int,
    numbercomponents::Int,
    dimension::Int;
    collocatedquadrature::Bool = false,
    mapping = nothing,
)
    # check inputs
    if numbernodes1d < 2
        throw(DomanError(numbernodes1d, "numbernodes1d must be greater than or equal to 2")) # COV_EXCL_LINE
    end
    if numberquadraturepoints1d < 1
        # COV_EXCL_START
        throw(
            DomanError(
                numberquadraturepoints1d,
                "numberquadraturepoints1d must be greater than or equal to 1",
            ),
        )
        # COV_EXCL_STOP
    end
    if dimension < 1 || dimension > 3
        throw(DomanError(dimension, "only 1D, 2D, or 3D bases are supported")) # COV_EXCL_LINE
    end

    # get nodes, quadrature points, and weights
    nodes1d = gausslobattoquadrature(numbernodes1d, false)
    quadraturepoints1d = []
    quadratureweights1d = []
    if collocatedquadrature
        quadraturepoints1d, quadratureweights1d =
            gausslobattoquadrature(numberquadraturepoints1d, true)
    else
        quadraturepoints1d, quadratureweights1d = gaussquadrature(numberquadraturepoints1d)
    end

    # build interpolation, gradient matrices
    interpolation1d, gradient1d = buildinterpolationandgradient(nodes1d, quadraturepoints1d)

    if !isnothing(mapping)
        _, g_prime = mapping
        gradient1d ./= g_prime.(quadraturepoints1d)
        nodes1d = transformquadrature(nodes1d, nothing, mapping)
        quadraturepoints1d, quadratureweights1d =
            transformquadrature(quadraturepoints1d, quadratureweights1d, mapping)
    end

    # use basic constructor
    return TensorBasis(
        numbernodes1d,
        numberquadraturepoints1d,
        numbercomponents,
        dimension,
        nodes1d,
        quadraturepoints1d,
        quadratureweights1d,
        interpolation1d,
        gradient1d,
    )
end

"""
```julia
TensorH1UniformBasis(
    numbernodes1d,
    numberquadraturepoints1d,
    numbercomponents,
    dimension,
)
```

Tensor product basis on uniformly spaced points with Gauss-Legendre quadrature points

# Arguments:
- `numbernodes1d`:             number of uniformly spaced nodes in 1 dimension
- `numberquadraturepoints1d`:  number of Gauss-Legendre quadrature points in 1 dimension
- `numbercomponents`:          number of components
- `dimension`:                 dimension of basis

# Returns:
- H1 Lagrange tensor product basis on uniformly spaced nodes object

# Example:
```jldoctest
# generate H1 Lagrange tensor product basis on uniformly spaced nodes
basis = TensorH1UniformBasis(4, 3, 2, 1);

# verify
println(basis)

# output
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 3
    numbercomponents: 2
    dimension: 1
```
"""
function TensorH1UniformBasis(
    numbernodes1d::Int,
    numberquadraturepoints1d::Int,
    numbercomponents::Int,
    dimension::Int,
)
    # check inputs
    if numbernodes1d < 2
        throw(DomanError(numbernodes1d, "numbernodes1d must be greater than or equal to 2")) # COV_EXCL_LINE
    end
    if numberquadraturepoints1d < 1
        # COV_EXCL_START
        throw(
            DomanError(
                numberquadraturepoints1d,
                "numberquadraturepoints1d must be greater than or equal to 1",
            ),
        )
        # COV_EXCL_STOP
    end
    if dimension < 1 || dimension > 3
        throw(DomanError(dimension, "only 1D, 2D, or 3D bases are supported")) # COV_EXCL_LINE
    end

    # get nodes, quadrature points, and weights
    nodes1d = [-1.0:(2.0/(numbernodes1d-1)):1.0...]
    quadraturepoints1d, quadratureweights1d = gaussquadrature(numberquadraturepoints1d)

    # build interpolation, gradient matrices
    interpolation1d, gradient1d = buildinterpolationandgradient(nodes1d, quadraturepoints1d)

    # use basic constructor
    return TensorBasis(
        numbernodes1d,
        numberquadraturepoints1d,
        numbercomponents,
        dimension,
        nodes1d,
        quadraturepoints1d,
        quadratureweights1d,
        interpolation1d,
        gradient1d,
    )
end

# ------------------------------------------------------------------------------
# -- marco element bases
# ------------------------------------------------------------------------------

"""
```julia
TensorMacroElementBasisFrom1D(
    numbernodes1d,
    numberquadraturepoints1d,
    numbercomponents,
    dimension,
    numberelements1d,
    basis1dmicro,
    overlapquadraturepoints = false,
)
```

Tensor product macro-element basis from 1d single element tensor product basis

# Arguments:
- `numbernodes1d`:             number of basis nodes in 1 dimension
- `numberquadraturepoints1d`:  number of quadrature points in 1 dimension
- `numbercomponents`:          number of components
- `dimension`:                 dimension of basis
- `numberelements1d`:          number of elements in macro-element
- `basis1dmicro`:              1d micro element basis to replicate 

# Keyword Arguments:
- `overlapquadraturepoints`:  Overlap quadrature points between elements, for prolongation
                                  default: false

# Returns:
- Tensor product macro-element basis object
"""
function TensorMacroElementBasisFrom1D(
    numbernodes1d::Int,
    numberquadraturepoints1d::Int,
    numbercomponents::Int,
    dimension::Int,
    numberelements1d::Int,
    basis1dmicro::TensorBasis;
    overlapquadraturepoints::Bool = false,
)
    if numberelements1d < 2
        # COV_EXCL_START
        throw(
            DomanError(numberelements1d, "macro-elements must contain at least 2 elements"),
        )
        # COV_EXCL_STOP
    end

    # compute dimensions
    numbernodes1dmacro = (numbernodes1d - 1) * numberelements1d + 1
    numberquadraturepoints1dmacro =
        overlapquadraturepoints ? (numberquadraturepoints1d - 1) * numberelements1d + 1 :
        numberquadraturepoints1d * numberelements1d

    # basis nodes
    lower = minimum(basis1dmicro.nodes1d)
    width = basis1dmicro.volume
    nodes1dmacro = zeros(numbernodes1dmacro)
    micronodes = (basis1dmicro.nodes1d .- lower) ./ numberelements1d
    for i = 1:numberelements1d
        nodes1dmacro[(i-1)*numbernodes1d-i+2:i*numbernodes1d-i+1] =
            micronodes .+ lower .+ width / numberelements1d * (i - 1)
    end

    # basis quadrature points and weights
    quadratureweights1dmacro = []
    if overlapquadraturepoints
        quadratureweights1dmacro = zeros(numberquadraturepoints1dmacro)
    else
        quadratureweights1dmacro =
            kron(ones(numberelements1d), basis1dmicro.quadratureweights1d)
    end
    quadraturepoints1dmacro = zeros(numberquadraturepoints1dmacro)
    microquadraturepoints = (basis1dmicro.quadraturepoints1d .- lower) ./ numberelements1d
    for i = 1:numberelements1d
        offset = overlapquadraturepoints ? -i + 1 : 0
        quadraturepoints1dmacro[(i-1)*numberquadraturepoints1d+offset+1:i*numberquadraturepoints1d+offset] =
            microquadraturepoints .+ lower .+ width / numberelements1d * (i - 1)
    end

    # basis operations
    interpolation1dmacro = spzeros(numberquadraturepoints1dmacro, numbernodes1dmacro)
    gradient1dmacro = spzeros(numberquadraturepoints1dmacro, numbernodes1dmacro)
    for i = 1:numberelements1d
        offset = overlapquadraturepoints ? -i + 1 : 0
        interpolation1dmacro[
            (i-1)*numberquadraturepoints1d+offset+1:i*numberquadraturepoints1d+offset,
            (i-1)*numbernodes1d-i+2:i*numbernodes1d-i+1,
        ] = basis1dmicro.interpolation1d
        gradient1dmacro[
            (i-1)*numberquadraturepoints1d+offset+1:i*numberquadraturepoints1d+offset,
            (i-1)*numbernodes1d-i+2:i*numbernodes1d-i+1,
        ] = basis1dmicro.gradient1d
    end

    # use basic constructor
    return TensorBasis(
        numbernodes1dmacro,
        numberquadraturepoints1dmacro,
        numbercomponents,
        dimension,
        nodes1dmacro,
        quadraturepoints1dmacro,
        quadratureweights1dmacro,
        interpolation1dmacro,
        gradient1dmacro,
        numberelements1d = numberelements1d,
    )
end

"""
```julia
TensorH1LagrangeMacroBasis(
    numbernodes1d,
    numberquadraturepoints1d,
    numbercomponents,
    dimension,
    numberelements1d,
    collocatedquadrature = false,
)
```

Tensor product macro-element basis on Gauss-Legendre-Lobatto points with
  Gauss-Legendre (default) or Gauss-Legendre-Lobatto quadrature points

# Arguments:
- `numbernodes1d`:                 number of Gauss-Legendre-Lobatto nodes in 1 dimension
- `numberquadraturepoints1d`:      number of quadrature points in 1 dimension
- `numbercomponents`:              number of components
- `dimension`:                     dimension of basis
- `numberelements1d`:              number of elements in macro-element

# Keyword Arguments:
- `collocatedquadrature = false`:  Gauss-Legendre or Gauss-Legendre-Lobatto quadrature points,
                                     default: false, Gauss-Legendre-Lobatto

# Returns:
- H1 Lagrange tensor product macro-element basis object

# Example:
```jldoctest
# generate H1 Lagrange tensor macro-element product basis
basis = TensorH1LagrangeMacroBasis(4, 4, 1, 2, 2);

# generate basis with Gauss-Legendre quadrature points
basis = TensorH1LagrangeMacroBasis(4, 4, 1, 2, 2; collocatedquadrature=true);

# verify
println(basis)

# output
macro-element tensor product basis:
    numbernodes1d: 7
    numberquadraturepoints1d: 8
    numbercomponents: 1
    numberelements1d: 2
    dimension: 2
```
"""
function TensorH1LagrangeMacroBasis(
    numbernodes1d::Int,
    numberquadraturepoints1d::Int,
    numbercomponents::Int,
    dimension::Int,
    numberelements1d::Int;
    collocatedquadrature::Bool = false,
)
    basis1dmicro = TensorH1LagrangeBasis(
        numbernodes1d,
        numberquadraturepoints1d,
        numbercomponents,
        1,
        collocatedquadrature = collocatedquadrature,
    )
    # use common constructor
    return TensorMacroElementBasisFrom1D(
        numbernodes1d,
        numberquadraturepoints1d,
        numbercomponents,
        dimension,
        numberelements1d,
        basis1dmicro,
    )
end

"""
```julia
TensorH1UniformMacroBasis(
    numbernodes1d,
    numberquadraturepoints1d,
    numbercomponents,
    dimension,
    numberelements1d,
)
```

Tensor product macro-element basis on uniformly points with Gauss-Legendre quadrature

# Arguments:
- `numbernodes1d`:             number of uniformly spaced nodes in 1 dimension
- `numberquadraturepoints1d`:  number of Gauss-Legendre quadrature points in 1 dimension
- `numbercomponents`:          number of components
- `dimension`:                 dimension of basis
- `numberelements1d`:          number of elements in macro-element

# Returns:
- H1 Lagrange tensor product macro-element basis on uniformly space nodes object

# Example:
```jldoctest
# generate H1 Lagrange tensor product macro-element basis on uniformly spaced nodes
basis = TensorH1UniformMacroBasis(4, 3, 1, 2, 2);

# verify
println(basis)

# output
macro-element tensor product basis:
    numbernodes1d: 7
    numberquadraturepoints1d: 6
    numbercomponents: 1
    numberelements1d: 2
    dimension: 2
```
"""
function TensorH1UniformMacroBasis(
    numbernodes1d::Int,
    numberquadraturepoints1d::Int,
    numbercomponents::Int,
    dimension::Int,
    numberelements1d::Int,
)
    basis1dmicro =
        TensorH1UniformBasis(numbernodes1d, numberquadraturepoints1d, numbercomponents, 1)
    # use common constructor
    return TensorMacroElementBasisFrom1D(
        numbernodes1d,
        numberquadraturepoints1d,
        numbercomponents,
        dimension,
        numberelements1d,
        basis1dmicro,
    )
end

# ------------------------------------------------------------------------------
# -- p multigrid interpolation bases
# ------------------------------------------------------------------------------

"""
```julia
TensorH1LagrangePProlongationBasis(
    numbercoarsenodes1d,
    numberfinenodes1d,
    numbercomponents,
    dimension,
)
```

Tensor product p-prolongation basis on Gauss-Legendre-Lobatto points

# Arguments:
- `numbercoarsenodes1d`:  number of coarse grid Gauss-Legendre-Lobatto nodes in 1 dimension
- `numberfinenodes1d`:    number of fine grid Gauss-Legendre-Lobatto nodes in 1 dimension
- `numbercomponents`:     number of components
- `dimension`:            dimension of basis

# Returns:
- H1 Lagrange tensor product p-prolongation basis object

# Example:
```jldoctest
# generate H1 Lagrange tensor product p-prolongation basis
basisctof = TensorH1LagrangePProlongationBasis(2, 3, 1, 2);

# verify
println(basisctof)

# output
tensor product basis:
    numbernodes1d: 2
    numberquadraturepoints1d: 3
    numbercomponents: 1
    dimension: 2
```
"""
function TensorH1LagrangePProlongationBasis(
    numbercoarsenodes1d::Int,
    numberfinenodes1d::Int,
    numbercomponents::Int,
    dimension::Int,
)
    return TensorH1LagrangeBasis(
        numbercoarsenodes1d,
        numberfinenodes1d,
        numbercomponents,
        dimension,
        collocatedquadrature = true,
    )
end

# ------------------------------------------------------------------------------
# -- h multigrid interpolation bases
# ------------------------------------------------------------------------------

"""
```julia
TensorHProlongationBasis(
    coarsenodes1d,
    finenodes1d,
    numbercomponents,
    dimension,
    numberfineelements1d,
)
```

Tensor product h-prolongation basis

# Arguments:
- `coarsenodes1d`:         coarse grid node coordinates in 1 dimension
- `finenodes1d`:           fine grid node coordinates in 1 dimension
- `numbercomponents`:      number of components
- `dimension`:             dimension of basis
- `numberfineelements1d`:  number of fine grid elements

# Returns:
- H1 Lagrange tensor product h-prolongation basis object
"""
function TensorHProlongationBasis(
    coarsenodes1d::AbstractArray{Float64,1},
    finenodes1d::AbstractArray{Float64,1},
    numbercomponents::Int,
    dimension::Int,
    numberfineelements1d::Int,
)
    # compute dimensions
    numbercoarsenodes1d = maximum(size(coarsenodes1d))
    numberfinenodes1d = maximum(size(finenodes1d))

    # form coarse to fine basis
    interpolation1d, gradient1d = buildinterpolationandgradient(coarsenodes1d, finenodes1d)
    quadratureweights1d = zeros(numberfinenodes1d)
    return TensorBasis(
        numbercoarsenodes1d,
        numberfinenodes1d,
        numbercomponents,
        dimension,
        coarsenodes1d,
        finenodes1d,
        quadratureweights1d,
        interpolation1d,
        gradient1d,
    )
end

"""
```julia
TensorH1LagrangeHProlongationBasis(
    numbernodes1d,
    numbercomponents,
    dimension,
    numberfineelements1d,
)
```

Tensor product h-prolongation basis on Gauss-Legendre-Lobatto points

# Arguments:
- `numbernodes1d`:         number of Gauss-Legendre-Lobatto nodes in 1 dimension per element
- `numbercomponents`:      number of components
- `dimension`:             dimension of basis
- `numberfineelements1d`:  number of fine grid elements

# Returns:
- H1 Lagrange tensor product h-prolongation basis object

# Example:
```jldoctest
# generate H1 Lagrange tensor product h-prolongation basis
basis = TensorH1LagrangeHProlongationBasis(4, 3, 2, 2);

# verify
println(basis)

# output
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 7
    numbercomponents: 3
    dimension: 2
```
"""
function TensorH1LagrangeHProlongationBasis(
    numbernodes1d::Int,
    numbercomponents::Int,
    dimension::Int,
    numberfineelements1d::Int,
)
    # generate nodes
    nodescoarse1d = gausslobattoquadrature(numbernodes1d, false)
    nodesfine1d = zeros((numbernodes1d - 1) * numberfineelements1d + 1)
    for i = 1:numberfineelements1d
        nodesfine1d[(i-1)*numbernodes1d-i+2:i*numbernodes1d-i+1] =
            nodescoarse1d ./ numberfineelements1d .+
            ((2 * (i - 1) + 1) / numberfineelements1d - 1)
    end

    # single coarse to multiple fine elements
    return TensorHProlongationBasis(
        nodescoarse1d,
        nodesfine1d,
        numbercomponents,
        dimension,
        numberfineelements1d,
    )
end

"""
```julia
TensorH1UniformHProlongationBasis(
    numbernodes1d,
    numbercomponents,
    dimension,
    numberfineelements1d,
)
```

Tensor product h-prolongation basis on uniformly spaced points

# Arguments:
- `numbernodes1d`:         number of uniformly spaced nodes per element
- `numbercomponents`:      number of components
- `dimension`:             dimension of basis
- `numberfineelements1d`:  number of fine grid elements

# Returns:
- H1 Lagrange tensor product h-prolongation basis on uniformly spaced nodes object

# Example:
```jldoctest
# generate H1 Lagrange tensor product h-prolongation basis on uniformly spaced nodes
basis = TensorH1UniformHProlongationBasis(4, 3, 2, 2);

# verify
println(basis)

# output
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 7
    numbercomponents: 3
    dimension: 2
```
"""
function TensorH1UniformHProlongationBasis(
    numbernodes1d::Int,
    numbercomponents::Int,
    dimension::Int,
    numberfineelements1d::Int,
)
    # generate nodes
    nodescoarse1d = [-1.0:(2.0/(numbernodes1d-1)):1.0...]
    nodesfine1d = [-1.0:(2.0/((numbernodes1d-1)*numberfineelements1d)):1.0...]
    for i = 1:numberfineelements1d
        nodesfine1d[(i-1)*numbernodes1d-i+2:i*numbernodes1d-i+1] =
            nodescoarse1d ./ numberfineelements1d .+
            ((2 * (i - 1) + 1) / numberfineelements1d - 1)
    end

    # single coarse to multiple fine elements
    return TensorHProlongationBasis(
        nodescoarse1d,
        nodesfine1d,
        numbercomponents,
        dimension,
        numberfineelements1d,
    )
end

"""
```julia
TensorH1LagrangeHProlongationMacroBasis(
    numbernodes1d,
    numbercomponents,
    dimension,
    numbercoarseelements1d,
    numberfineelements1d,
)
```

Tensor product macro-element h-prolongation basis on Gauss-Legendre-Lobatto points

# Arguments:
- `numbernodes1d`:           number of Gauss-Legendre-Lobatto nodes in 1 dimension per element
- `numbercomponents`:        number of components
- `dimension`:               dimension of basis
- `numbercoarseelements1d`:  number of coarse grid elements in macro-element
- `numberfineelements1d`:    number of fine grid elements in macro-element

# Returns:
- H1 Lagrange tensor product h-prolongation macro-element basis object

# Example:
```jldoctest
# generate H1 Lagrange tensor product h-prolongation macro-element basis
basis = TensorH1LagrangeHProlongationMacroBasis(4, 1, 2, 2, 4);

# verify
println(basis)

# output
macro-element tensor product basis:
    numbernodes1d: 7
    numberquadraturepoints1d: 13
    numbercomponents: 1
    numberelements1d: 2
    dimension: 2
```
"""
function TensorH1LagrangeHProlongationMacroBasis(
    numbernodes1d::Int,
    numbercomponents::Int,
    dimension::Int,
    numbercoarseelements1d::Int,
    numberfineelements1d::Int,
)
    # validate inputs
    if numberfineelements1d % numbercoarseelements1d != 0
        # COV_EXCL_START
        throw(
            DomanError(
                numberfineelements1d,
                "numberfineelements1d must be a multiple of numbercoarseelements1d",
            ),
        )
        # COV_EXCL_STOP
    end

    # single coarse to multiple fine elements
    scale = Int(numberfineelements1d / numbercoarseelements1d)
    hprolongationbasis1dmicro =
        TensorH1LagrangeHProlongationBasis(numbernodes1d, numbercomponents, 1, scale)

    # use common constructor
    return TensorMacroElementBasisFrom1D(
        numbernodes1d,
        (numbernodes1d - 1) * scale + 1,
        numbercomponents,
        dimension,
        numbercoarseelements1d,
        hprolongationbasis1dmicro,
        overlapquadraturepoints = true,
    )
end

"""
```
TensorH1UniformHProlongationMacroBasis(
    numbernodes1d,
    numbercomponents,
    dimension,
    numbercoarseelements1d,
    numberfineelements1d,
)
```

Tensor product macro-element h-prolongation basis on uniformly spaced points

# Arguments:
- `numbernodes1d`:           number of uniformly spaced nodes per element
- `numbercomponents`:        number of components
- `dimension`:               dimension of basis
- `numbercoarseelements1d`:  number of coarse grid elements in macro-element
- `numberfineelements1d`:    number of fine grid elements in macro-element

# Returns:
- H1 Lagrange tensor product h-prolongation macro-element basis on uniformly spaced nodes object

# Example:
```jldoctest
# generate H1 Lagrange tensor product h-prolongation macro-element basis on uniformly spaced nodes
basis = TensorH1UniformHProlongationMacroBasis(4, 1, 2, 2, 4);

# verify
println(basis)

# output
macro-element tensor product basis:
    numbernodes1d: 7
    numberquadraturepoints1d: 13
    numbercomponents: 1
    numberelements1d: 2
    dimension: 2
```
"""
function TensorH1UniformHProlongationMacroBasis(
    numbernodes1d::Int,
    numbercomponents::Int,
    dimension::Int,
    numbercoarseelements1d::Int,
    numberfineelements1d::Int,
)
    # validate inputs
    if numberfineelements1d % numbercoarseelements1d != 0
        # COV_EXCL_START
        throw(
            DomanError(
                numberfineelements1d,
                "numberfineelements1d must be a multiple of numbercoarseelements1d",
            ),
        )
        # COV_EXCL_STOP
    end

    # single coarse to multiple fine elements
    scale = Int(numberfineelements1d / numbercoarseelements1d)
    hprolongationbasis1dmicro =
        TensorH1UniformHProlongationBasis(numbernodes1d, numbercomponents, 1, scale)

    # use common constructor
    return TensorMacroElementBasisFrom1D(
        numbernodes1d,
        (numbernodes1d - 1) * scale + 1,
        numbercomponents,
        dimension,
        numbercoarseelements1d,
        hprolongationbasis1dmicro,
        overlapquadraturepoints = true,
    )
end

# ------------------------------------------------------------------------------
