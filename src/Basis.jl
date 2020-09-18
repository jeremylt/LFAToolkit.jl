# ---------------------------------------------------------------------------------------------------------------------
# Finite element bases
# ---------------------------------------------------------------------------------------------------------------------

"""
Finite element basis for function spaces and test spaces
"""
abstract type Basis end

# ---------------------------------------------------------------------------------------------------------------------
# Basic basis types
# ---------------------------------------------------------------------------------------------------------------------

"""
```julia
TensorBasis(
    p1d,
    q1d,
    dimension,
    nodes1d,
    quadraturepoints1d,
    quadratureweights1d,
    interpolation1d,
    gradient1d
)
```

Tensor product basis

# Arguments:
- `p1d`:                 number of nodes in 1 dimension
- `q1d`:                 number of quadrature points in 1 dimension
- `dimension`:           dimension of the basis
- `nodes1d`:             coordinates of the nodes in 1 dimension
- `quadraturepoints1d`:  coordinates of the quadrature points in 1 dimension
- `quadratureweights1d`: quadrature weights in 1 dimension
- `interpolation1d`:     interpolation matrix from nodes to quadrature points in 1 dimension
- `gradient1d`:          gradient matrix from nodes to quadrature points in 1 dimension

# Returns:
- Tensor product basis object
"""
mutable struct TensorBasis <: Basis
    # never changed
    p1d::Int
    q1d::Int
    dimension::Int
    nodes1d::Array{Float64,1}
    quadraturepoints1d::Array{Float64,1}
    quadratureweights1d::Array{Float64,1}
    interpolation1d::Array{Float64,2}
    gradient1d::Array{Float64,2}

    # empty until assembled
    nodes::Array{Float64}
    quadraturepoints::Array{Float64}
    quadratureweights::Array{Float64,1}
    interpolation::Array{Float64,2}
    gradient::Array{Float64,2}

    # constructor
    TensorBasis(
        p1d,
        q1d,
        dimension,
        nodes1d,
        quadraturepoints1d,
        quadratureweights1d,
        interpolation1d,
        gradient1d,
    ) = (
        # validity checking
        if p1d <= 0
            error("p1d must be at least 1") # COV_EXCL_LINE
        end;
        if q1d <= 0
            error("q1d must be at least 1") # COV_EXCL_LINE
        end;
        if dimension <= 0
            error("dimension must be at least 1") # COV_EXCL_LINE
        end;
        if length(nodes1d) != p1d
            error("must include p1d nodes") # COV_EXCL_LINE
        end;
        if length(quadraturepoints1d) != q1d
            error("must include q1d quadrature points") # COV_EXCL_LINE
        end;
        if length(quadratureweights1d) != q1d
            error("must include q1d quadrature weights") # COV_EXCL_LINE
        end;
        if size(interpolation1d) != (q1d, p1d)
            error("interpolation matrix must have dimensions (q1d, p1d)") # COV_EXCL_LINE
        end;
        if size(gradient1d) != (q1d, p1d)
            error("gradient matrix must have dimensions (q1d, p1d)") # COV_EXCL_LINE
        end;

        # constructor
        new(
            p1d,
            q1d,
            dimension,
            nodes1d,
            quadraturepoints1d,
            quadratureweights1d,
            interpolation1d,
            gradient1d,
        )
    )
end

"""
```julia
NonTensorBasis(
    p,
    q,
    dimension,
    nodes,
    quadraturepoints,
    quadratureweights,
    interpolation,
    gradient
)
```
Non-tensor basis

# Arguments:
- `p`:                 number of nodes 
- `q`:                 number of quadrature points
- `dimension`:         dimension of the basis
- `nodes`:             coordinates of the nodes
- `quadraturepoints`:  coordinates of the quadrature points
- `quadratureweights`: quadrature weights
- `interpolation`:     interpolation matrix from nodes to quadrature points
- `gradient`:          gradient matrix from nodes to quadrature points

# Returns:
- Non-tensor product basis object
"""
mutable struct NonTensorBasis <: Basis
    # never changed
    p::Int
    q::Int
    dimension::Int
    nodes::Array{Float64}
    quadraturepoints::Array{Float64}
    quadratureweights::Array{Float64,1}
    interpolation::Array{Float64,2}
    gradient::Array{Float64,2}

    # constructor
    NonTensorBasis(
        p,
        q,
        dimension,
        nodes,
        quadraturepoints,
        quadratureweights,
        interpolation,
        gradient,
    ) = (
        # validity checking
        if p <= 0
            error("p must be at least 1") # COV_EXCL_LINE
        end;
        if q <= 0
            error("q must be at least 1") # COV_EXCL_LINE
        end;
        if dimension <= 0
            error("dimension must be at least 1") # COV_EXCL_LINE
        end;
        if size(nodes) != (p, dimension)
            error("must include (p, dimension) nodal coordinates") # COV_EXCL_LINE
        end;
        if length(quadraturepoints) != (q, dimension)
            error("must include (q, dimension) quadrature points") # COV_EXCL_LINE
        end;
        if length(quadratureweights) != q
            error("must include q quadrature weights") # COV_EXCL_LINE
        end;
        if size(interpolation) != (q, p)
            error("interpolation matrix must have dimensions (q, p)") # COV_EXCL_LINE
        end;
        if size(gradient) != (q * dimension, p)
            error("gradient matrix must have dimensions (q*dimension, p)") # COV_EXCL_LINE
        end;

        # constructor
        new(
            p,
            q,
            dimension,
            nodes,
            quadraturepoints,
            quadratureweights,
            interpolation,
            gradient,
        )
    )
end

# ---------------------------------------------------------------------------------------------------------------------
# Utility functions for generating polynomial bases
# ---------------------------------------------------------------------------------------------------------------------

"""
```julia
gaussquadrature(q)
```

Construct a Gauss-Legendre quadrature

# Arguments:
- `q`: number of Gauss-Legendre points

# Returns:
- Gauss-Legendre quadrature points and weights

# Example:
```jldoctest
# generate Gauss-Legendre points and weights
quadraturepoints, quadratureweights = LFAToolkit.gaussquadrature(5);

# verify
truepoints = [
    -sqrt(5 + 2*sqrt(10/7))/3,
    -sqrt(5 - 2*sqrt(10/7))/3,
    0.0,
    sqrt(5 - 2*sqrt(10/7))/3,
    sqrt(5 + 2*sqrt(10/7))/3
];
trueweights = [
    (322-13*sqrt(70))/900,
    (322+13*sqrt(70))/900,
    128/225,
    (322+13*sqrt(70))/900,
    (322-13*sqrt(70))/900
];

diff = truepoints - quadraturepoints;
@assert abs(max(diff...)) < 1e-15

diff = trueweights - quadratureweights;
@assert abs(max(diff...)) < 1e-15

# output

```
"""
function gaussquadrature(q::Int)
    quadraturepoints = zeros(Float64, q)
    quadratureweights = zeros(Float64, q)

    if q < 1
        throw(DomanError(basis.dimension, "q must be greater than or equal to 1")) # COV_EXCL_LINE
    end

    # build qref1d, qweight1d
    for i = 0:floor(Int, q / 2)
        # guess
        xi = cos(pi * (2 * i + 1.0) / (2 * q))

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
        itr = 0
        while itr < 100 && abs(p2) > 1e-15
            p0 = 1.0
            p1 = xi
            for j = 2:q
                p2 = ((2 * j - 1.0) * xi * p1 - (j - 1.0) * p0) / j
                p0 = p1
                p1 = p2
            end
            dp2 = (xi * p2 - p0) * q / (xi * xi - 1.0)
            xi = xi - p2 / dp2
            itr += 1
        end

        # save xi, wi
        quadraturepoints[i+1] = -xi
        quadraturepoints[q-i] = xi
        wi = 2.0 / ((1.0 - xi * xi) * dp2 * dp2)
        quadratureweights[i+1] = wi
        quadratureweights[q-i] = wi
    end

    # return
    return quadraturepoints, quadratureweights
end

"""
```julia
lobattoquadrature(q, weights)
```

Construct a Gauss-Lobatto quadrature

# Arguments:
- `q`:       number of Gauss-Lobatto points
- `weights`: boolean flag indicating if quadrature weights are desired

# Returns:
- Gauss-Lobatto quadrature points or points and weights

# Example:
```jldoctest
# generate Gauss-Lobatto points
quadraturepoints = LFAToolkit.lobattoquadrature(5, false);

# verify
truepoints = [-1.0, -sqrt(3/7), 0.0, sqrt(3/7), 1.0];

diff = truepoints - quadraturepoints;
@assert abs(max(diff...)) < 1e-15

# generate Gauss-Lobatto points and weights
quadraturepoints, quadratureweights = LFAToolkit.lobattoquadrature(5, true);

# verify
trueweights = [1/10, 49/90, 32/45, 49/90, 1/10];

diff = trueweights - quadratureweights;
@assert abs(max(diff...)) < 1e-15

# output

```
"""
function lobattoquadrature(q::Int, weights::Bool)
    quadraturepoints = zeros(Float64, q)
    quadratureweights = zeros(Float64, q)

    if q < 2
        throw(DomanError(basis.dimension, "q must be greater than or equal to 2")) # COV_EXCL_LINE
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
        xi = cos(pi * i / (q - 1.0))

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
        itr = 0
        while itr < 100 && abs(dp2) > 1e-15
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
            itr += 1
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

    # return
    if weights
        return quadraturepoints, quadratureweights
    else
        return quadraturepoints
    end
end

# ---------------------------------------------------------------------------------------------------------------------
# User utility constructors
# ---------------------------------------------------------------------------------------------------------------------

"""
```julia
TensorH1LagrangeBasis(p1d, q1d, dimension)
```

Tensor product basis on Gauss-Lobatto points with Gauss-Legendre quadrature

# Arguments:
- `p1d`:       number of Gauss-Lobatto nodes
- `q1d`:       number of Gauss-Legendre quadrature points
- `dimension`: dimension of basis

# Returns:
- H1 Lagrange tensor product basis object

# Example:
```jldoctest
# generate H1 Lagrange tensor product basis
basis = TensorH1LagrangeBasis(4, 3, 2);

# verify
@assert basis.p1d == 4
@assert basis.q1d == 3
@assert basis.dimension == 2

# output

```
"""
function TensorH1LagrangeBasis(p1d::Int, q1d::Int, dimension::Int)
    # check inputs
    if p1d <= 2
        throw(DomanError(p1d, "p1d must be greater than or equal to 2")) # COV_EXCL_LINE
    end
    if q1d <= 1
        throw(DomanError(p1d, "q1d must be greater than or equal to 1")) # COV_EXCL_LINE
    end
    if dimension < 1 || dimension > 3
        throw(DomanError(dimension, "only 1D, 2D, or 3D bases are supported")) # COV_EXCL_LINE
    end

    # get nodes, quadrature points, and weights
    nodes1d = lobattoquadrature(p1d, false)
    quadraturepoints1d, quadratureweights1d = gaussquadrature(q1d)

    # build interpolation, gradient matrices
    # Fornberg, 1998
    interpolation1d = zeros(Float64, q1d, p1d)
    gradient1d = zeros(Float64, q1d, p1d)
    for i = 1:q1d
        c1 = 1.0
        c3 = nodes1d[1] - quadraturepoints1d[i]
        interpolation1d[i, 1] = 1.0
        for j = 2:p1d
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

    # use basic constructor
    return TensorBasis(
        p1d,
        q1d,
        dimension,
        nodes1d,
        quadraturepoints1d,
        quadratureweights1d,
        interpolation1d,
        gradient1d,
    )
end

# ---------------------------------------------------------------------------------------------------------------------
# Return basis information about the bases
# ---------------------------------------------------------------------------------------------------------------------

"""
```julia
getnumbernodes(basis)
```

Get the number of nodes for the basis

# Arguments:
- `basis`: basis to compute number of nodes

# Returns:
- Integer number of basis nodes

# Example:
```jldoctest
# get number of nodes for basis
basis = TensorH1LagrangeBasis(4, 3, 2);
numbernodes = LFAToolkit.getnumbernodes(basis);
numbernodes = basis.p; # either syntax works

# verify
@assert numbernodes == 4^2

# output

```
"""
function getnumbernodes(basis::NonTensorBasis)
    return basis.p
end

function getnumbernodes(basis::TensorBasis)
    return basis.p1d^basis.dimension
end

"""
```julia
getnodes(basis)
```

Get nodes for basis

# Returns:
- Basis nodes array

# Arguments:
- `basis`: basis to compute nodes

# Example:
```jldoctest
# test for all supported dimensions
for dimension in 1:3
    # get basis quadrature weights
    basis = TensorH1LagrangeBasis(4, 3, dimension);
    nodes = LFAToolkit.getnodes(basis);
    nodes = basis.nodes; # either syntax works

    # verify
    truenodes1d = [-1, -sqrt(1/5), sqrt(1/5), 1];
    truenodes = [];
    if dimension == 1
        truenodes = truenodes1d;
    elseif dimension == 2
        truenodes =
            transpose(hcat([[[x, y] for x in truenodes1d, y in truenodes1d]...]...));
    elseif dimension == 3
        truenodes = transpose(hcat([[
            [x, y, z] for x in truenodes1d, y in truenodes1d, z in truenodes1d
        ]...]...));
    end

    diff = truenodes - nodes;
    @assert abs(max(max(diff...)...)) < 1e-15
end
    
# output

```
"""
function getnodes(basis::NonTensorBasis)
    return basis.nodes
end

function getnodes(basis::TensorBasis)
    # assembled if needed
    if !isdefined(basis, :nodes)
        nodes = []
        if basis.dimension == 1
            # 1D
            nodes = basis.nodes1d
        elseif basis.dimension == 2
            # 2D
            nodes =
                transpose(hcat([[[x, y] for x in basis.nodes1d, y in basis.nodes1d]...]...))
        elseif basis.dimension == 3
            # 3D
            nodes = transpose(hcat([[
                [x, y, z] for x in basis.nodes1d, y in basis.nodes1d, z in basis.nodes1d
            ]...]...))
        else
            throw(DomanError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
        end
        basis.nodes = nodes
    end

    # return
    return getfield(basis, :nodes)
end

"""
```julia
getnumberquadraturepoints(basis)
```

Get the number of quadrature points for the basis

# Arguments:
- `basis`: basis to compute number of quadrature points

# Returns:
- Integer number of basis quadrature points

# Example:
```jldoctest
# get number of quadrature points for basis
basis = TensorH1LagrangeBasis(4, 3, 2);
quadraturepoints = LFAToolkit.getnumberquadraturepoints(basis);
quadraturepoints = basis.q; # either syntax works
    
# verify
@assert quadraturepoints == 3^2
    
# output

```
"""
function getnumberquadraturepoints(basis::NonTensorBasis)
    return basis.q
end

function getnumberquadraturepoints(basis::TensorBasis)
    return basis.q1d^basis.dimension
end

"""
```julia
getquadraturepoints(basis)
```

Get quadrature points for basis

# Returns:
- Basis quadrature points array

# Arguments:
- `basis`: basis to compute quadrature points

# Example:
```jldoctest
# test for all supported dimensions
for dimension in 1:3
    # get basis quadrature weights
    basis = TensorH1LagrangeBasis(4, 3, dimension);
    quadraturepoints = LFAToolkit.getquadraturepoints(basis);
    quadraturepoints = basis.quadraturepoints; # either syntax works

    # verify
    truequadraturepoints1d = [-sqrt(3/5), 0, sqrt(3/5)];
    truequadraturepoints = [];
    if dimension == 1
        truequadraturepoints = truequadraturepoints1d;
    elseif dimension == 2
        truequadraturepoints = transpose(hcat([[
            [x, y] for x in truequadraturepoints1d, y in truequadraturepoints1d
        ]...]...));
    elseif dimension == 3
        truequadraturepoints = transpose(hcat([[
            [x, y, z]
            for
            x in truequadraturepoints1d,
            y in truequadraturepoints1d, z in truequadraturepoints1d
        ]...]...));
    end

    diff = truequadraturepoints - quadraturepoints;
    @assert abs(max(max(diff...)...)) < 1e-15
end
    
# output

```
"""
function getquadraturepoints(basis::NonTensorBasis)
    return basis.quadraturepoints
end

function getquadraturepoints(basis::TensorBasis)
    # assembled if needed
    if !isdefined(basis, :quadraturepoints)
        quadraturepoints = []
        if basis.dimension == 1
            # 1D
            quadraturepoints = basis.quadraturepoints1d
        elseif basis.dimension == 2
            # 2D
            quadraturepoints = transpose(hcat([[
                [x, y] for x in basis.quadraturepoints1d, y in basis.quadraturepoints1d
            ]...]...))
        elseif basis.dimension == 3
            # 3D
            quadraturepoints = transpose(hcat([[
                [x, y, z]
                for
                x in basis.quadraturepoints1d,
                y in basis.quadraturepoints1d, z in basis.quadraturepoints1d
            ]...]...))
        else
            throw(DomanError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
        end
        basis.quadraturepoints = quadraturepoints
    end

    # return
    return getfield(basis, :quadraturepoints)
end

"""
```julia
getquadratureweights(basis)
```

Get full quadrature weights vector for basis

# Returns:
- Basis quadrature weights vector

# Arguments:
- `basis`: basis to compute quadrature weights

# Example:
```jldoctest
# test for all supported dimensions
for dimension in 1:3
    # get basis quadrature weights
    basis = TensorH1LagrangeBasis(4, 3, dimension);
    quadratureweights = LFAToolkit.getquadratureweights(basis);
    quadratureweights = basis.quadratureweights; # either syntax works

    # verify
    trueweights1d = [5/9, 8/9, 5/9];
    trueweights = [];
    if dimension == 1
        trueweights = trueweights1d;
    elseif dimension == 2
        trueweights = kron(trueweights1d, trueweights1d);
    elseif dimension == 3
        trueweights = kron(trueweights1d, trueweights1d, trueweights1d);
    end

    diff = trueweights - quadratureweights;
    @assert abs(max(diff...)) < 1e-15
end
    
# output

```
"""
function getquadratureweights(basis::NonTensorBasis)
    return basis.quadratureweights
end

function getquadratureweights(basis::TensorBasis)
    # assemble if needed
    if !isdefined(basis, :quadratureweights)
        quadratureweights = []
        if basis.dimension == 1
            # 1D
            quadratureweights = basis.quadratureweights1d
        elseif basis.dimension == 2
            # 2D
            quadratureweights = kron(basis.quadratureweights1d, basis.quadratureweights1d)
        elseif basis.dimension == 3
            # 3D
            quadratureweights = kron(
                basis.quadratureweights1d,
                basis.quadratureweights1d,
                basis.quadratureweights1d,
            )
        else
            throw(DomanError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
        end
        basis.quadratureweights = quadratureweights
    end

    # return
    return getfield(basis, :quadratureweights)
end

# ---------------------------------------------------------------------------------------------------------------------
# Basis functions for constructing stencils
# ---------------------------------------------------------------------------------------------------------------------

"""
```julia
getinterpolation(basis)
```

Get full interpolation matrix for basis

# Arguments:
- `basis`: basis to compute interpolation matrix

# Returns:
- Basis interpolation matrix

# Example:
```jldoctest
# test for all supported dimensions
for dimension in 1:3
    # get basis interpolation matrix
    basis = TensorH1LagrangeBasis(4, 3, dimension);
    interpolation = LFAToolkit.getinterpolation(basis);
    interpolation = basis.interpolation; # either syntax works

    # verify
    for i in 1:3^dimension
        total = sum(interpolation[i, :]);
        @assert abs(total - 1.0) < 1e-15
    end
end

# output

```
"""
function getinterpolation(basis::NonTensorBasis)
    return basis.interpolation
end

function getinterpolation(basis::TensorBasis)
    # assemble if needed
    if !isdefined(basis, :interpolation)
        interpolation = []
        if basis.dimension == 1
            # 1D
            interpolation = basis.interpolation1d
        elseif basis.dimension == 2
            # 2D
            interpolation = kron(basis.interpolation1d, basis.interpolation1d)
        elseif basis.dimension == 3
            # 3D
            interpolation =
                kron(basis.interpolation1d, basis.interpolation1d, basis.interpolation1d)
        else
            throw(DomanError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
        end
        basis.interpolation = interpolation
    end

    # return
    return getfield(basis, :interpolation)
end

"""
```julia
getgradient(basis)
```

Get full gradient matrix for basis

# Arguments:
- `basis`: basis to compute gradient matrix

# Returns:
- Basis gradient matrix

# Example:
```jldoctest
# test for all supported dimensions
for dimension in 1:3
    # get basis gradient matrix
    basis = TensorH1LagrangeBasis(4, 3, dimension);
    gradient = LFAToolkit.getgradient(basis);
    gradient = basis.gradient; # either syntax works

    # verify
    for i in 1:dimension*3^dimension
        total = sum(gradient[i, :]);
        @assert abs(total) < 1e-14
    end
end

# output

```
"""
function getgradient(basis::NonTensorBasis)
    return basis.gradient
end

function getgradient(basis::TensorBasis)
    # assemble if needed
    if !isdefined(basis, :gradient)
        gradient = []
        if basis.dimension == 1
            # 1D
            gradient = basis.gradient1d
        elseif basis.dimension == 2
            # 2D
            gradient = [
                kron(basis.gradient1d, basis.interpolation1d)
                kron(basis.interpolation1d, basis.gradient1d)
            ]
        elseif basis.dimension == 3
            # 3D
            gradient = [
                kron(basis.gradient1d, basis.interpolation1d, basis.interpolation1d)
                kron(basis.interpolation1d, basis.gradient1d, basis.interpolation1d)
                kron(basis.interpolation1d, basis.interpolation1d, basis.gradient1d)
            ]
        else
            throw(DomanError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
        end
        basis.gradient = gradient
    end

    # return
    return getfield(basis, :gradient)
end

# ---------------------------------------------------------------------------------------------------------------------
# get/set property
# ---------------------------------------------------------------------------------------------------------------------

function Base.getproperty(basis::TensorBasis, f::Symbol)
    if f == :p
        return getnumbernodes(basis)
    elseif f == :q
        return getnumberquadraturepoints(basis)
    elseif f == :nodes
        return getnodes(basis)
    elseif f == :quadratureweights
        return getquadratureweights(basis)
    elseif f == :quadraturepoints
        return getquadraturepoints(basis)
    elseif f == :interpolation
        return getinterpolation(basis)
    elseif f == :gradient
        return getgradient(basis)
    else
        return getfield(basis, f)
    end
end

# ---------------------------------------------------------------------------------------------------------------------
