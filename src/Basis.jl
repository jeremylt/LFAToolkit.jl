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
Tensor product basis
"""
struct TensorBasis <: Basis
    p1d::Int
    q1d::Int
    dimension::Int
    numbercomponents::Int
    nodes1d::Array{Float64,1}
    quadraturepoints1d::Array{Float64,1}
    quadratureweights1d::Array{Float64,1}
    interpolation1d::Array{Float64,2}
    gradient1d::Array{Float64,2}
end

"""
Non-tensor basis
"""
struct NonTensorBasis <: Basis
    p::Int
    q::Int
    dimension::Int
    numbercomponents::Int
    nodes::Array{Float64}
    quadraturepoints::Array{Float64}
    quadratureweights::Array{Float64,1}
    interpolation::Array{Float64,2}
    gradient::Array{Float64,2}
end

# ---------------------------------------------------------------------------------------------------------------------
# Utility functions for generating polynomial bases
# ---------------------------------------------------------------------------------------------------------------------

"""
    gaussquadrature()

Construct a Gauss-Legendre quadrature

```jldoctest
quadraturepoints, quadratureweights = LFAToolkit.gaussquadrature(5);

# verify
truepoints = [-sqrt(5 + 2*sqrt(10/7))/3, -sqrt(5 - 2*sqrt(10/7))/3, 0.0, sqrt(5 - 2*sqrt(10/7))/3, sqrt(5 + 2*sqrt(10/7))/3];
trueweights = [(322-13*sqrt(70))/900, (322+13*sqrt(70))/900, 128/225, (322+13*sqrt(70))/900, (322-13*sqrt(70))/900];

diff = truepoints - quadraturepoints;
if abs(max(diff...)) > 1e-15
    println("Incorrect quadrature points");
end

diff = trueweights - quadratureweights;
if abs(abs(max(diff...))) > 1e-15
    println("Incorrect quadrature weights");
end

# output

```
"""
function gaussquadrature(q::Int)
    quadraturepoints = zeros(Float64, q)
    quadratureweights = zeros(Float64, q)

    if q < 2
        throw(DomanError(basis.dimension, "q must be greater than or equal to 2")) # COV_EXCL_LINE
    end

    # build qrefid, qweight1d
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
    lobattoquadrature()

Construct a Gauss-Lobatto quadrature

```jldoctest
quadraturepoints = LFAToolkit.lobattoquadrature(5, false);

# verify
truepoints = [-1.0, -sqrt(3/7), 0.0, sqrt(3/7), 1.0];

diff = truepoints - quadraturepoints;
if abs(max(diff...)) > 1e-15
    println("Incorrect quadrature points");
end

# output

```
   
```jldoctest
quadraturepoints, quadratureweights = LFAToolkit.lobattoquadrature(5, true);

# verify
truepoints = [-1.0, -sqrt(3/7), 0.0, sqrt(3/7), 1.0];
trueweights = [1/10, 49/90, 32/45, 49/90, 1/10];

diff = truepoints - quadraturepoints;
if abs(max(diff...)) > 1e-15
    println("Incorrect quadrature points");
end

diff = trueweights - quadratureweights;
if abs(abs(max(diff...))) > 1e-15
    println("Incorrect quadrature weights");
end

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

    # build qrefid, qweight1d
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
    TensorH1LagrangeBasis()

Tensor product basis on Gauss-Lobatto points with Gauss-Legendre quadrature

```jldoctest
basis = TensorH1LagrangeBasis(4, 3, 2, 1);

# verify
if basis.p1d != 4
    println("Incorrect P1d");
end
if basis.q1d != 3
    println("Incorrect Q1d");
end
if basis.dimension != 2
    println("Incorrect dimension");
end
if basis.numbercomponents != 1
    println("Incorrect number of components");
end

# output

```
"""
function TensorH1LagrangeBasis(p1d::Int, q1d::Int, dimension::Int, numbercomponents::Int)
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
        numbercomponents,
        nodes1d,
        quadraturepoints1d,
        quadratureweights1d,
        interpolation1d,
        gradient1d,
    )
end

# ---------------------------------------------------------------------------------------------------------------------
# Basis functions for constructing stensils
# ---------------------------------------------------------------------------------------------------------------------

"""
    getinterpolation()

Get full interpolation matrix for basis

```jldoctest
basis = TensorH1LagrangeBasis(4, 3, 1, 1);
interpolation = LFAToolkit.getinterpolation(basis);

# verify
for i in 1:3
    sum = 0.0
    for j = 1:4
        sum += interpolation[i, j];
    end
    if abs(sum - 1.0) > 1e-15
        println("Incorrect interpolation matrix")
    end
end

# output

```

```jldoctest
basis = TensorH1LagrangeBasis(4, 3, 2, 1);
interpolation = LFAToolkit.getinterpolation(basis);

# verify
for i in 1:3^2
    sum = 0.0
    for j = 1:4^2
        sum += interpolation[i, j];
    end
    if abs(sum - 1.0) > 1e-15
        println("Incorrect interpolation matrix")
    end
end

# output

```

```jldoctest
basis = TensorH1LagrangeBasis(4, 3, 3, 1);
interpolation = LFAToolkit.getinterpolation(basis);

# verify
for i in 1:3^3
    sum = 0.0
    for j = 1:4^3
        sum += interpolation[i, j];
    end
    if abs(sum - 1.0) > 1e-15
        println("Incorrect interpolation matrix")
    end
end

# output

```
"""
function getinterpolation(basis::NonTensorBasis)
    return basis.interpolation
end

function getinterpolation(basis::TensorBasis)
    if basis.dimension == 1
        # 1D
        return basis.interpolation1d
    elseif basis.dimension == 2
        # 2D
        return kron(basis.interpolation1d, basis.interpolation1d)
    elseif basis.dimension == 3
        # 3D
        return kron(basis.interpolation1d, basis.interpolation1d, basis.interpolation1d)
    end
    throw(DomanError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
end

"""
    getgradient()

Get full gradient matrix for basis

```jldoctest
basis = TensorH1LagrangeBasis(4, 3, 1, 1);
gradient = LFAToolkit.getgradient(basis);

# verify
for i in 1:3
    sum = 0.0
    for j = 1:4
        sum += gradient[i, j];
    end
    if abs(sum) > 1e-15
        println("Incorrect gradent matrix")
    end
end

# output

```

```jldoctest
basis = TensorH1LagrangeBasis(4, 3, 2, 1);
gradient = LFAToolkit.getgradient(basis);

# verify
for d in 1:2, i in 1:3^2
    sum = 0.0
    for j = 1:4^2
        sum += gradient[d, i, j];
    end
    if abs(sum) > 1e-15
        println("Incorrect gradient matrix")
    end
end

# output

```

```jldoctest
basis = TensorH1LagrangeBasis(4, 3, 3, 1);
gradient = LFAToolkit.getgradient(basis);

# verify
for d in 1:3, i in 1:3^3
    sum = 0.0
    for j = 1:4^3
        sum += gradient[d, i, j];
    end
    if abs(sum) > 1e-14
        println("Incorrect gradient matrix")
    end
end

# output

```
"""
function getgradient(basis::NonTensorBasis)
    return basis.gradient
end

function getgradient(basis::TensorBasis)
    if basis.dimension == 1
        # 1D
        return basis.gradient1d
    else
        numbernodes = basis.p1d^basis.dimension
        numberqpoints = basis.q1d^basis.dimension
        gradient = ones(basis.dimension, numberqpoints, numbernodes)
        if basis.dimension == 2
            # 2D
            gradient[1, :, :] = kron(basis.gradient1d, basis.interpolation1d)
            gradient[2, :, :] = kron(basis.interpolation1d, basis.gradient1d)
        elseif basis.dimension == 3
            # 3D
            gradient[1, :, :] =
                kron(basis.gradient1d, basis.interpolation1d, basis.interpolation1d)
            gradient[2, :, :] =
                kron(basis.interpolation1d, basis.gradient1d, basis.interpolation1d)
            gradient[3, :, :] =
                kron(basis.interpolation1d, basis.interpolation1d, basis.gradient1d)
        end
        return gradient
    end
    throw(DomanError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
end

"""
    getquadratureweights()

Get full quadrature weights vector for basis

```jldoctest
basis = TensorH1LagrangeBasis(4, 3, 1, 1);
quadratureweights = LFAToolkit.getquadratureweights(basis);

# verify
trueweights = [5/9, 8/9, 5/9];

diff = trueweights - quadratureweights;
if abs(abs(max(diff...))) > 1e-15
    println("Incorrect quadrature weights");
end

# output

```

```jldoctest
basis = TensorH1LagrangeBasis(4, 3, 2, 1);
quadratureweights = LFAToolkit.getquadratureweights(basis);

# verify
trueweights1d = [5/9, 8/9, 5/9];
trueweights = kron(trueweights1d, trueweights1d);

diff = trueweights - quadratureweights;
if abs(abs(max(diff...))) > 1e-15
    println("Incorrect quadrature weights");
end
    
# output

```

```jldoctest
basis = TensorH1LagrangeBasis(4, 3, 3, 1);
quadratureweights = LFAToolkit.getquadratureweights(basis);

# verify
trueweights1d = [5/9, 8/9, 5/9];
trueweights = kron(trueweights1d, trueweights1d, trueweights1d);

diff = trueweights - quadratureweights;
if abs(abs(max(diff...))) > 1e-15
    println("Incorrect quadrature weights");
end
    
# output

```
"""
function getquadratureweights(basis::NonTensorBasis)
    return basis.quadratureweights
end

function getquadratureweights(basis::TensorBasis)
    if basis.dimension == 1
        # 1D
        return basis.quadratureweights1d
    elseif basis.dimension == 2
        # 2D
        return kron(basis.quadratureweights1d, basis.quadratureweights1d)
    elseif basis.dimension == 3
        # 3D
        return kron(
            basis.quadratureweights1d,
            basis.quadratureweights1d,
            basis.quadratureweights1d,
        )
    end
    throw(DomanError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
end

# ---------------------------------------------------------------------------------------------------------------------
