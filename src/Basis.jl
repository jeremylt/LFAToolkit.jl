# ------------------------------------------------------------------------------
# finite element bases
# ------------------------------------------------------------------------------

"""
Finite element basis for function spaces and test spaces
"""
abstract type AbstractBasis end

# ------------------------------------------------------------------------------
# basis types
# ------------------------------------------------------------------------------

"""
```julia
TensorBasis(
    numbernodes1d,
    numberquadraturepoints1d,
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
- `numbernodes1d`:            number of nodes in 1 dimension
- `numberquadraturepoints1d`: number of quadrature points in 1 dimension
- `dimension`:                dimension of the basis
- `nodes1d`:                  coordinates of the nodes in 1 dimension
- `quadraturepoints1d`:       coordinates of the quadrature points in 1
                                  dimension
- `quadratureweights1d`:      quadrature weights in 1 dimension
- `interpolation1d`:          interpolation matrix from nodes to quadrature
                                  points in 1 dimension
- `gradient1d`:               gradient matrix from nodes to quadrature points in
                                  1 dimension

# Returns:
- Tensor product basis object
"""
mutable struct TensorBasis <: AbstractBasis
    # data never changed
    numbernodes1d::Int
    numberquadratuepoints1d::Int
    dimension::Int
    nodes1d::AbstractArray{Float64,1}
    quadraturepoints1d::AbstractArray{Float64,1}
    quadratureweights1d::AbstractArray{Float64,1}
    interpolation1d::AbstractArray{Float64,2}
    gradient1d::AbstractArray{Float64,2}
    volume::Float64
    numberelements1d::Int

    # data empty until assembled
    nodes::AbstractArray{Float64}
    quadraturepoints::AbstractArray{Float64}
    quadratureweights::AbstractArray{Float64,1}
    interpolation::AbstractArray{Float64,2}
    gradient::AbstractArray{Float64,2}
    numbermodes::Int
    modemap::AbstractArray{Int,1}

    # inner constructor
    TensorBasis(
        numbernodes1d::Int,
        numberquadraturepoints1d::Int,
        dimension::Int,
        nodes1d::AbstractArray{Float64,1},
        quadraturepoints1d::AbstractArray{Float64,1},
        quadratureweights1d::AbstractArray{Float64,1},
        interpolation1d::AbstractArray{Float64,2},
        gradient1d::AbstractArray{Float64,2};
        numberelements1d::Int = 1,
    ) = (
        # validity checking
        if numbernodes1d < 1
            error("p1d must be at least 1") # COV_EXCL_LINE
        end;
        if numberquadraturepoints1d < 1
            error("q1d must be at least 1") # COV_EXCL_LINE
        end;
        if dimension < 1
            error("dimension must be at least 1") # COV_EXCL_LINE
        end;
        if length(nodes1d) != numbernodes1d
            error("must include numbernodes1d nodes") # COV_EXCL_LINE
        end;
        if length(quadraturepoints1d) != numberquadraturepoints1d
            error("must include numberquadraturepoints1d quadrature points") # COV_EXCL_LINE
        end;
        if length(quadratureweights1d) != numberquadraturepoints1d
            error("must include numberquadraturepoints1d quadrature weights") # COV_EXCL_LINE
        end;
        if size(interpolation1d) != (numberquadraturepoints1d, numbernodes1d)
            # COV_EXCL_START
            error(
                "interpolation matrix must have dimensions (numberquadraturepoints1d, numbernodes1d)",
            )
            # COV_EXCL_STOP
        end;
        if size(gradient1d) != (numberquadraturepoints1d, numbernodes1d)
            # COV_EXCL_START
            error(
                "gradient matrix must have dimensions (numberquadraturepoints1d, numbernodes1d)",
            ) # COV_EXCL_STOP
        end;

        # constructor
        new(
            numbernodes1d,
            numberquadraturepoints1d,
            dimension,
            nodes1d,
            quadraturepoints1d,
            quadratureweights1d,
            interpolation1d,
            gradient1d,
            (max(nodes1d...) - min(nodes1d...))^dimension,
            numberelements1d,
        )
    )
end

# printing
# COV_EXCL_START
function Base.show(io::IO, basis::TensorBasis)
    print(
        io,
        basis.numberelements1d == 1 ? "" : "macro element ",
        "tensor product basis:\n    numbernodes1d: ",
        basis.numbernodes1d,
        "\n    numberquadraturepoints1d: ",
        basis.numberquadratuepoints1d,
    )
    if basis.numberelements1d != 1
        print(io, "\n    numberelements1d: ", basis.numberelements1d)
    end
    print(io, "\n    dimension: ", basis.dimension)
end
# COV_EXCL_STOP

"""
```julia
NonTensorBasis(
    numbernodes,
    numberquadraturepoints,
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
- `numbernodes`:            number of nodes 
- `numberquadraturepoints`: number of quadrature points
- `dimension`:              dimension of the basis
- `nodes`:                  coordinates of the nodes
- `quadraturepoints`:       coordinates of the quadrature points
- `quadratureweights`:      quadrature weights
- `interpolation`:          interpolation matrix from nodes to quadrature points
- `gradient`:               gradient matrix from nodes to quadrature points

# Returns:
- Non-tensor product basis object
"""
mutable struct NonTensorBasis <: AbstractBasis
    # data never changed
    numbernodes::Int
    numberquadraturepoints::Int
    dimension::Int
    nodes::AbstractArray{Float64}
    quadraturepoints::AbstractArray{Float64,1}
    quadratureweights::AbstractArray{Float64,1}
    interpolation::AbstractArray{Float64,2}
    gradient::AbstractArray{Float64,2}
    volume::Float64
    numbermodes::Int
    modemap::AbstractArray{Int,1}
    numberelements::Int

    # inner constructor
    NonTensorBasis(
        numbernodes::Int,
        numberquadraturepoints::Int,
        dimension::Int,
        nodes::Int,
        quadraturepoints::AbstractArray{Float64},
        quadratureweights::AbstractArray{Float64,1},
        interpolation::AbstractArray{Float64,2},
        gradient::AbstractArray{Float64,2};
        numberelements::Int = 1,
    ) = (
        # validity checking
        if numbernodes < 1
            error("number of nodes must be at least 1") # COV_EXCL_LINE
        end;
        if numberquadraturepoints < 1
            error("number of quadrature points must be at least 1") # COV_EXCL_LINE
        end;
        if dimension < 1
            error("dimension must be at least 1") # COV_EXCL_LINE
        end;
        if size(nodes) != (numbernodes, dimension)
            error("must include (numbernodes, dimension) nodal coordinates") # COV_EXCL_LINE
        end;
        if length(quadraturepoints) != (numberquadraturepoints, dimension)
            error("must include (numberquadraturepoints, dimension) quadrature points") # COV_EXCL_LINE
        end;
        if length(quadratureweights) != numberquadraturepoints
            error("must include sufficient quadrature weights") # COV_EXCL_LINE
        end;
        if size(interpolation) != (numberquadraturepoints, numbernodes)
            # COV_EXCL_START
            error(
                "interpolation matrix must have dimensions (numberquadraturepoints, numbernodes)",
            )
            # COV_EXCL_STOP
        end;
        if size(gradient) != (q*dimension, numbernodes)
            # COV_EXCL_START
            error(
                "gradient matrix must have dimensions (numberquadraturepoints*dimension, numbernodes)",
            )
            # COV_EXCL_STOP
        end;
        if length(modemap) != numbernodes
            error("must map the modes for each basis node") # COV_EXCL_LINE
        end;

        # compute volume
        volume = 1;
        for d = 1:dimension
            volume *= (max(nodes[:, d]...) - min(nodes[:, d]...))
        end;

        # constructor
        new(
            numbernodes,
            numberquadraturepoints,
            dimension,
            nodes,
            quadraturepoints,
            quadratureweights,
            interpolation,
            gradient,
            volume,
            max(modemap...),
            modemap,
            numberelements,
        )
    )
end

# printing
# COV_EXCL_START
function Base.show(io::IO, basis::NonTensorBasis)
    print(
        io,
        basis.numberelements == 1 ? "" : "macro element ",
        "non-tensor product basis:\n    numbernodes: ",
        basis.numbernodes,
        "\n    numberquadraturepoints: ",
        basis.numberquadratuepoints,
    )
    if basis.numberelements != 1
        print(io, "\n    numberelements: ", basis.numberelements)
    end
    print(io, "\n    dimension: ", basis.dimension)
end
# COV_EXCL_STOP

# ------------------------------------------------------------------------------
# utility functions for generating polynomial bases
# ------------------------------------------------------------------------------

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
function gaussquadrature(q::Int)
    quadraturepoints = zeros(Float64, q)
    quadratureweights = zeros(Float64, q)

    if q < 1
        throw(DomanError(basis.dimension, "q must be greater than or equal to 1")) # COV_EXCL_LINE
    end

    # build qref1d, qweight1d
    for i = 0:floor(Int, q/2)
        # guess
        xi = cos(pi*(2*i + 1.0)/(2*q))

        # Pn(xi)
        p0 = 1.0
        p1 = xi
        p2 = 0.0
        for j = 2:q
            p2 = ((2*j - 1.0)*xi*p1 - (j - 1.0)*p0)/j
            p0 = p1
            p1 = p2
        end

        # first Newton step
        dp2 = (xi*p2 - p0)*q/(xi*xi - 1.0)
        xi = xi - p2/dp2

        # Newton to convergence
        itr = 0
        while itr < 100 && abs(p2) > 1e-15
            p0 = 1.0
            p1 = xi
            for j = 2:q
                p2 = ((2*j - 1.0)*xi*p1 - (j - 1.0)*p0)/j
                p0 = p1
                p1 = p2
            end
            dp2 = (xi*p2 - p0)*q/(xi*xi - 1.0)
            xi = xi - p2/dp2
            itr += 1
        end

        # save xi, wi
        quadraturepoints[i+1] = -xi
        quadraturepoints[q-i] = xi
        wi = 2.0/((1.0 - xi*xi)*dp2*dp2)
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
truepoints = [-1.0, -√(3/7), 0.0, √(3/7), 1.0];
@assert truepoints ≈ quadraturepoints

# generate Gauss-Lobatto points and weights
quadraturepoints, quadratureweights = LFAToolkit.lobattoquadrature(5, true);

# verify
trueweights = [1/10, 49/90, 32/45, 49/90, 1/10];
@assert trueweights ≈ quadratureweights

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
        wi = 2.0/(q*(q - 1.0))
        quadratureweights[1] = wi
        quadratureweights[q] = wi
    end

    # build qref1d, qweight1d
    for i = 1:floor(Int, (q - 1)/2)
        # guess
        xi = cos(pi*i/(q - 1.0))

        # Pn(xi)
        p0 = 1.0
        p1 = xi
        p2 = 0.0
        for j = 2:q-1
            p2 = ((2*j - 1.0)*xi*p1 - (j - 1.0)*p0)/j
            p0 = p1
            p1 = p2
        end

        # first Newton step
        dp2 = (xi*p2 - p0)*q/(xi*xi - 1.0)
        d2p2 = (2*xi*dp2 - q*(q - 1.0)*p2)/(1.0 - xi*xi)
        xi = xi - dp2/d2p2

        # Newton to convergence
        itr = 0
        while itr < 100 && abs(dp2) > 1e-15
            p0 = 1.0
            p1 = xi
            for j = 2:q-1
                p2 = ((2*j - 1.0)*xi*p1 - (j - 1.0)*p0)/j
                p0 = p1
                p1 = p2
            end
            dp2 = (xi*p2 - p0)*q/(xi*xi - 1.0)
            d2p2 = (2*xi*dp2 - q*(q - 1.0)*p2)/(1.0 - xi*xi)
            xi = xi - dp2/d2p2
            itr += 1
        end

        # save xi, wi
        quadraturepoints[i+1] = -xi
        quadraturepoints[q-i] = xi
        if weights
            wi = 2.0/(q*(q - 1.0)*p2*p2)
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

"""
```julia
buildinterpolationandgradient(
    nodes,
    quadraturepoints,
)
```

Build one dimensional interpolation and gradient matrices, from Fornberg 1998

# Arguments:
- `nodes`:            1d basis nodes
- `quadraturepoints`: 1d basis quadrature points

# Returns:
- One dimensional interpolation and gradient matrices

# Example:
```jldoctest
# get nodes, quadrature points, and weights
numbernodes = 3;
numberquadraturepoints = 4;
nodes = LFAToolkit.lobattoquadrature(numbernodes, false);
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
                    gradient1d[i, j] = c1*(interpolation1d[i, k] - c4*gradient1d[i, k])/c2
                    interpolation1d[i, j] = -c1*c4*interpolation1d[i, k]/c2
                end
                gradient1d[i, k] = (c3*gradient1d[i, k] - interpolation1d[i, k])/dx
                interpolation1d[i, k] = c3*interpolation1d[i, k]/dx
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
    dimension,
    lagrangequadrature = false,
)
```

Tensor product basis on Gauss-Lobatto points with Gauss-Legendre quadrature

# Arguments:
- `numbernodes1d`:            number of Gauss-Lobatto nodes
- `numberquadraturepoints1d`: number of Gauss-Legendre quadrature points
- `dimension`:                dimension of basis
- `lagrangequadrature=false`: Gauss-Lagrange or Gauss-Lobatto quadrature points,
                                  default: Gauss-Lobatto

# Returns:
- H1 Lagrange tensor product basis object

# Example:
```jldoctest
# generate H1 Lagrange tensor product basis
basis = TensorH1LagrangeBasis(4, 4, 2);

# generate basis with Lagrange quadrature points
basis = TensorH1LagrangeBasis(4, 4, 2, lagrangequadrature=true);

# verify
println(basis)

# output
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    dimension: 2
```
"""
function TensorH1LagrangeBasis(
    numbernodes1d::Int,
    numberquadraturepoints1d::Int,
    dimension::Int;
    lagrangequadrature::Bool = false,
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
    nodes1d = lobattoquadrature(numbernodes1d, false)
    quadraturepoints1d = []
    quadratureweights1d = []
    if lagrangequadrature
        quadraturepoints1d, quadratureweights1d =
            lobattoquadrature(numberquadraturepoints1d, true)
    else
        quadraturepoints1d, quadratureweights1d = gaussquadrature(numberquadraturepoints1d)
    end

    # build interpolation, gradient matrices
    interpolation1d, gradient1d = buildinterpolationandgradient(nodes1d, quadraturepoints1d)

    # use basic constructor
    return TensorBasis(
        numbernodes1d,
        numberquadraturepoints1d,
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
    dimension,
)
```

Tensor product basis on uniformly points with Gauss-Legendre quadrature

# Arguments:
- `numbernodes1d`:            number of uniformly spaced nodes
- `numberquadraturepoints1d`: number of Gauss-Legendre quadrature points
- `dimension`:                dimension of basis

# Returns:
- H1 uniformly spaced tensor product basis object

# Example:
```jldoctest
# generate H1 uniformly spaced tensor product basis
basis = TensorH1UniformBasis(4, 3, 2);

# verify
println(basis)

# output
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 3
    dimension: 2
```
"""
function TensorH1UniformBasis(
    numbernodes1d::Int,
    numberquadraturepoints1d::Int,
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
    dimension,
    numberelements1d,
    basis1dmicro,
    overlapquadraturepoints = false,
)
```

Tensor product macro element basis from 1d single element tensor product basis

# Arguments:
- `numbernodes1d`:            number of basis nodes
- `numberquadraturepoints1d`: number of quadrature points
- `dimension`:                dimension of basis
- `numberelements1d`:         number of elements in macro element
- `basis1dmicro`:             1d micro element basis to replicate 
    
# Returns:
- Tensor product macro element basis object
"""
function TensorMacroElementBasisFrom1D(
    numbernodes1d::Int,
    numberquadraturepoints1d::Int,
    dimension::Int,
    numberelements1d::Int,
    basis1dmicro::TensorBasis;
    overlapquadraturepoints::Bool = false,
)
    if numberelements1d < 2
        throw(
            DomanError(numberelements1d, "macro elements must contain at least 2 elements"),
        ) # COV_EXCL_LINE
    end

    # compute dimensions
    numbernodes1dmacro = (numbernodes1d - 1)*numberelements1d + 1
    numberquadraturepoints1dmacro =
        overlapquadraturepoints ? (numberquadraturepoints1d - 1)*numberelements1d + 1 :
        numberquadraturepoints1d*numberelements1d

    # basis nodes
    width = basis1dmicro.volume
    nodes1dmacro = zeros(numbernodes1dmacro)
    nodes1dmacro[1:numbernodes1d] = basis1dmicro.nodes1d
    for i = 2:numberelements1d
        nodes1dmacro[(i-1)*numbernodes1d-i+2:i*numbernodes1d-i+1] =
            basis1dmicro.nodes1d .+ width*(i - 1)
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
    quadraturepoints1dmacro[1:numberquadraturepoints1d] = basis1dmicro.quadraturepoints1d
    for i = 2:numberelements1d
        offset = overlapquadraturepoints ? -i + 1 : 0
        quadraturepoints1dmacro[(i-1)*numberquadraturepoints1d+offset+1:i*numberquadraturepoints1d+offset] =
            basis1dmicro.quadraturepoints1d .+ width*(i - 1)
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
    dimension,
    numberelements1d,
    lagrangequadrature,
)
```

Tensor product macro element basis on Gauss-Lobatto points with Gauss-Legendre quadrature

# Arguments:
- `numbernodes1d`:            number of Gauss-Lobatto nodes
- `numberquadraturepoints1d`: number of Gauss-Legendre quadrature points
- `dimension`:                dimension of basis
- `numberelements1d`:         number of elements in macro element
- `lagrangequadrature=false`: Gauss-Lagrange or Gauss-Lobatto quadrature points,
                                  default: Gauss-Lobatto

# Returns:
- H1 Lagrange tensor product macro element basis object

# Example:
```jldoctest
# generate H1 Lagrange tensor macro element product basis
basis = TensorH1LagrangeMacroBasis(4, 4, 2, 2);

# generate basis with Lagrange quadrature points
basis = TensorH1LagrangeMacroBasis(4, 4, 2, 2, lagrangequadrature=true);

# verify
println(basis)

# output
macro element tensor product basis:
    numbernodes1d: 7
    numberquadraturepoints1d: 8
    numberelements1d: 2
    dimension: 2
```
"""
function TensorH1LagrangeMacroBasis(
    numbernodes1d::Int,
    numberquadraturepoints1d::Int,
    dimension::Int,
    numberelements1d::Int;
    lagrangequadrature::Bool = false,
)
    basis1dmicro = TensorH1LagrangeBasis(
        numbernodes1d,
        numberquadraturepoints1d,
        1,
        lagrangequadrature = lagrangequadrature,
    )
    # use common constructor
    return TensorMacroElementBasisFrom1D(
        numbernodes1d,
        numberquadraturepoints1d,
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
    dimension,
    numberelements1d,
)
```

Tensor product macro element basis on uniformly points with Gauss-Legendre quadrature

# Arguments:
- `numbernodes1d`:            number of uniformly spaced nodes
- `numberquadraturepoints1d`: number of Gauss-Legendre quadrature points
- `dimension`:                dimension of basis
- `numberelements1d`:         number of elements in macro element

# Returns:
- H1 uniformly spaced tensor product macro element basis object

# Example:
```jldoctest
# generate H1 uniformly spaced tensor product macro element basis
basis = TensorH1UniformMacroBasis(4, 3, 2, 2);

# verify
println(basis)

# output
macro element tensor product basis:
    numbernodes1d: 7
    numberquadraturepoints1d: 6
    numberelements1d: 2
    dimension: 2
```
"""
function TensorH1UniformMacroBasis(
    numbernodes1d::Int,
    numberquadraturepoints1d::Int,
    dimension::Int,
    numberelements1d::Int,
)
    basis1dmicro = TensorH1UniformBasis(numbernodes1d, numberquadraturepoints1d, 1)
    # use common constructor
    return TensorMacroElementBasisFrom1D(
        numbernodes1d,
        numberquadraturepoints1d,
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
    dimension,
)
```

Tensor product p prolongation basis on Gauss-Lobatto points

# Arguments:
- `numbercoarsenodes1d`: number of coarse grid Gauss-Lobatto nodes
- `numberfinenodes1d`:   number of fine grid Gauss-Lobatto nodes
- `dimension`:           dimension of basis

# Returns:
- H1 Lagrange tensor product basis object

# Example:
```jldoctest
# generate H1 Lagrange tensor product basis
basisctof = TensorH1LagrangePProlongationBasis(2, 3, 2);

# verify
println(basisctof)

# output
tensor product basis:
    numbernodes1d: 2
    numberquadraturepoints1d: 3
    dimension: 2
```
"""
function TensorH1LagrangePProlongationBasis(
    numbercoarsenodes1d::Int,
    numberfinenodes1d::Int,
    dimension::Int,
)
    return TensorH1LagrangeBasis(
        numbercoarsenodes1d,
        numberfinenodes1d,
        dimension,
        lagrangequadrature = true,
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
    dimension,
    numberfineelements1d,
)
```

Tensor product h prolongation basis

# Arguments:
- `numbernodes1d`:            number of Gauss-Lobatto nodes per element
- `dimension`:                dimension of basis
- `numberfineelements1d`:     number of fine grid elements

# Returns:
- H1 tensor product h prolongation basis object
"""
function TensorHProlongationBasis(
    coarsenodes1d::AbstractArray{Float64,1},
    finenodes1d::AbstractArray{Float64,1},
    dimension::Int,
    numberfineelements1d::Int,
)
    # compute dimensions
    numbercoarsenodes1d = max(size(coarsenodes1d)...)
    numberfinenodes1d = max(size(finenodes1d)...)

    # form coarse to fine basis
    interpolation1d, gradient1d = buildinterpolationandgradient(coarsenodes1d, finenodes1d)
    quadratureweights1d = zeros(numberfinenodes1d)
    return TensorBasis(
        numbercoarsenodes1d,
        numberfinenodes1d,
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
    dimension,
    numberfineelements1d,
)
```

Tensor product h prolongation basis on Gauss-Lobatto points

# Arguments:
- `numbernodes1d`:            number of Gauss-Lobatto nodes per element
- `dimension`:                dimension of basis
- `numberfineelements1d`:     number of fine grid elements

# Returns:
- H1 Gauss-Lobatto tensor product h prolongation basis object

# Example:
```jldoctest
# generate H1 Gauss-Lobatto tensor product h prolongation basis
basis = TensorH1LagrangeHProlongationBasis(4, 3, 2);

# verify
println(basis)

# output
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 7
    dimension: 3
```
"""
function TensorH1LagrangeHProlongationBasis(
    numbernodes1d::Int,
    dimension::Int,
    numberfineelements1d::Int,
)
    # generate nodes
    nodescoarse1d = lobattoquadrature(numbernodes1d, false)
    nodesfine1d = zeros((numbernodes1d - 1)*numberfineelements1d + 1)
    for i = 1:numberfineelements1d
        nodesfine1d[(i-1)*numbernodes1d-i+2:i*numbernodes1d-i+1] =
            nodescoarse1d./numberfineelements1d .+
            ((2*(i - 1) + 1)/numberfineelements1d - 1)
    end

    # single coarse to multiple fine elements
    return TensorHProlongationBasis(
        nodescoarse1d,
        nodesfine1d,
        dimension,
        numberfineelements1d,
    )
end

"""
```julia
TensorH1UniformHProlongationBasis(
    numbernodes1d,
    dimension,
    numberfineelements1d,
)
```

Tensor product h prolongation basis on uniformly spaced points

# Arguments:
- `numbernodes1d`:            number of uniformly spaced nodes per element
- `dimension`:                dimension of basis
- `numberfineelements1d`:     number of fine grid elements

# Returns:
- H1 uniformly spaced tensor product h prolongation basis object

# Example:
```jldoctest
# generate H1 uniformly spaced tensor product h prolongation basis
basis = TensorH1UniformHProlongationBasis(4, 3, 2);

# verify
println(basis)

# output
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 7
    dimension: 3
```
"""
function TensorH1UniformHProlongationBasis(
    numbernodes1d::Int,
    dimension::Int,
    numberfineelements1d::Int,
)
    # generate nodes
    nodescoarse1d = [-1.0:(2.0/(numbernodes1d-1)):1.0...]
    nodesfine1d = [-1.0:(2.0/((numbernodes1d-1)*numberfineelements1d)):1.0...]
    for i = 1:numberfineelements1d
        nodesfine1d[(i-1)*numbernodes1d-i+2:i*numbernodes1d-i+1] =
            nodescoarse1d./numberfineelements1d .+
            ((2*(i - 1) + 1)/numberfineelements1d - 1)
    end

    # single coarse to multiple fine elements
    return TensorHProlongationBasis(
        nodescoarse1d,
        nodesfine1d,
        dimension,
        numberfineelements1d,
    )
end

"""
```julia
TensorH1LagrangeHProlongationMacroBasis(
    numbernodes1d,
    dimension,
    numbercoarseelements1d,
    numberfineelements1d,
)
```

Tensor product macro element h prolongation basis on Gauss-Lobatto points

# Arguments:
- `numbernodes1d`:            number of Gauss-Lobatto nodes per element
- `dimension`:                dimension of basis
- `numbercoarseelements1d`:   number of coarse grid elements in macro element
- `numberfineelements1d`:     number of fine grid elements in macro element

# Returns:
- H1 Gauss-Lobatto tensor product h prolongation macro element basis object

# Example:
```jldoctest
# generate H1 Gauss-Lobatto tensor product h prolongation macro element basis
basis = TensorH1LagrangeHProlongationMacroBasis(4, 2, 2, 4);

# verify
println(basis)

# output
macro element tensor product basis:
    numbernodes1d: 7
    numberquadraturepoints1d: 13
    numberelements1d: 2
    dimension: 2
```
"""
function TensorH1LagrangeHProlongationMacroBasis(
    numbernodes1d::Int,
    dimension::Int,
    numbercoarseelements1d::Int,
    numberfineelements1d::Int,
)
    # validate inputs
    if numberfineelements1d%numbercoarseelements1d != 0
        throw(
            DomanError(
                numberfineelements1d,
                "numberfineelements1d must be a multiple of numbercoarseelements1d",
            ),
        ) # COV_EXCL_LINE
    end

    # single coarse to multiple fine elements
    scale = Int(numberfineelements1d/numbercoarseelements1d)
    hprolongationbasis1dmicro = TensorH1LagrangeHProlongationBasis(numbernodes1d, 1, scale)

    # use common constructor
    return TensorMacroElementBasisFrom1D(
        numbernodes1d,
        (numbernodes1d - 1)*scale + 1,
        dimension,
        numbercoarseelements1d,
        hprolongationbasis1dmicro,
        overlapquadraturepoints = true,
    )
end

"""
TensorH1UniformHProlongationMacroBasis(
    numbernodes1d,
    dimension,
    numbercoarseelements1d,
    numberfineelements1d,
)
```

Tensor product macro element h prolongation basis on uniformly spaced points

# Arguments:
- `numbernodes1d`:            number of uniformly spaced nodes per element
- `dimension`:                dimension of basis
- `numbercoarseelements1d`:   number of coarse grid elements in macro element
- `numberfineelements1d`:     number of fine grid elements in macro element

# Returns:
- H1 uniformly spaced tensor product h prolongation macro element basis object

# Example:
```jldoctest
# generate H1 uniformly spaced tensor product h prolongation macro element basis
basis = TensorH1UniformHProlongationMacroBasis(4, 2, 2, 4);

# verify
println(basis)

# output
macro element tensor product basis:
    numbernodes1d: 7
    numberquadraturepoints1d: 13
    numberelements1d: 2
    dimension: 2
```
"""
function TensorH1UniformHProlongationMacroBasis(
    numbernodes1d::Int,
    dimension::Int,
    numbercoarseelements1d::Int,
    numberfineelements1d::Int,
)
    # validate inputs
    if numberfineelements1d%numbercoarseelements1d != 0
        throw(
            DomanError(
                numberfineelements1d,
                "numberfineelements1d must be a multiple of numbercoarseelements1d",
            ),
        ) # COV_EXCL_LINE
    end

    # single coarse to multiple fine elements
    scale = Int(numberfineelements1d/numbercoarseelements1d)
    hprolongationbasis1dmicro = TensorH1UniformHProlongationBasis(numbernodes1d, 1, scale)

    # use common constructor
    return TensorMacroElementBasisFrom1D(
        numbernodes1d,
        (numbernodes1d - 1)*scale + 1,
        dimension,
        numbercoarseelements1d,
        hprolongationbasis1dmicro,
        overlapquadraturepoints = true,
    )
end

# ------------------------------------------------------------------------------
# basis properties
# ------------------------------------------------------------------------------

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

# note: either syntax works
numbernodes = LFAToolkit.getnumbernodes(basis);
numbernodes = basis.numbernodes;

# verify
@assert numbernodes == 4^2

# output

```
"""
function getnumbernodes(basis::TensorBasis)
    return basis.numbernodes1d^basis.dimension
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

    # note: either syntax works
    nodes = LFAToolkit.getnodes(basis);
    nodes = basis.nodes;

    # verify
    truenodes1d = [-1, -√(1/5), √(1/5), 1];
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

    @assert truenodes ≈ nodes
end
    
# output

```
"""
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
            nodes = transpose(
                hcat(
                    [
                        [
                            [x, y, z] for x in basis.nodes1d, y in basis.nodes1d,
                            z in basis.nodes1d
                        ]...,
                    ]...,
                ),
            )
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

# note: either syntax works
numberquadraturepoints = LFAToolkit.getnumberquadraturepoints(basis);
numberquadraturepoints = basis.numberquadraturepoints;
    
# verify
@assert numberquadraturepoints == 3^2
    
# output

```
"""
function getnumberquadraturepoints(basis::TensorBasis)
    return basis.numberquadratuepoints1d^basis.dimension
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

    # note: either syntax works
    quadraturepoints = LFAToolkit.getquadraturepoints(basis);
    quadraturepoints = basis.quadraturepoints;

    # verify
    truequadraturepoints1d = [-√(3/5), 0, √(3/5)];
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

    @assert truequadraturepoints ≈ quadraturepoints
end
    
# output

```
"""
function getquadraturepoints(basis::TensorBasis)
    # assembled if needed
    if !isdefined(basis, :quadraturepoints)
        quadraturepoints = []
        if basis.dimension == 1
            # 1D
            quadraturepoints = basis.quadraturepoints1d
        elseif basis.dimension == 2
            # 2D
            quadraturepoints = transpose(
                hcat(
                    [
                        [
                            [x, y] for x in basis.quadraturepoints1d,
                            y in basis.quadraturepoints1d
                        ]...,
                    ]...,
                ),
            )
        elseif basis.dimension == 3
            # 3D
            quadraturepoints = transpose(
                hcat(
                    [
                        [
                            [x, y, z] for x in basis.quadraturepoints1d,
                            y in basis.quadraturepoints1d, z in basis.quadraturepoints1d
                        ]...,
                    ]...,
                ),
            )
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

    # note: either syntax works
    quadratureweights = LFAToolkit.getquadratureweights(basis);
    quadratureweights = basis.quadratureweights;

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

    @assert trueweights ≈ quadratureweights
end
    
# output

```
"""
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

    # note: either syntax works
    interpolation = LFAToolkit.getinterpolation(basis);
    interpolation = basis.interpolation;

    # verify
    for i in 1:3^dimension
        total = sum(interpolation[i, :]);
        @assert total ≈ 1.0
    end
end

# output

```
"""
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

    # note: either syntax works
    gradient = LFAToolkit.getgradient(basis);
    gradient = basis.gradient;

    # verify
    for i in 1:dimension*3^dimension
        total = sum(gradient[i, :]);
        @assert abs(total) < 1e-14
    end
end

# output

```
"""
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

"""
```julia
getnumbermodes(basis)
```

Get number of modes for basis

# Arguments:
- `basis`: basis to compute number of modes

# Returns:
- Number of modes for basis

# Example:
```jldoctest
# test for all supported dimensions
for dimension in 1:3
    # get number of basis modes
    basis = TensorH1LagrangeBasis(4, 3, dimension);

    # note: either syntax works
    numbermodes = LFAToolkit.getnumbermodes(basis);
    numbermodes = basis.numbermodes;

    # verify
    @assert numbermodes == 3^dimension
end

# output

```
"""
function getnumbermodes(basis::TensorBasis)
    # assemble if needed
    if !isdefined(basis, :modemap)
        basis.numbermodes = max(basis.modemap...)
    end

    # return
    return getfield(basis, :numbermodes)
end

"""
```julia
getmodemap(basis)
```

Get mode mapping vector for basis

# Arguments:
- `basis`: basis to compute mode map vector

# Returns:
- Basis mode map vector

# Example:
```jldoctest
# test for all supported dimensions
for dimension in 1:3
    # get mode map vector
    basis = TensorH1LagrangeBasis(4, 3, dimension);

    # note: either syntax works
    modemap = LFAToolkit.getmodemap(basis);
    modemap = basis.modemap;

    # verify
    truemodemap1d = [1, 2, 3, 1];
    truemodemap = [];
    if dimension == 1
        truemodemap = truemodemap1d;
    elseif dimension == 2
        truemodemap = [[
            i + (j - 1)*3 for i in truemodemap1d, j in truemodemap1d
        ]...];
    elseif dimension == 3
        truemodemap = [[
            i +
            (j - 1)*3 +
            (k - 1)*3^2
            for i in truemodemap1d, j in truemodemap1d, k in truemodemap1d
        ]...];
    end

    @assert truemodemap == modemap
end

# output

```
"""
function getmodemap(basis::TensorBasis)
    # assemble if needed
    if !isdefined(basis, :modemap)
        modemap1d = [1:basis.numbernodes1d;]
        modemap1d[end] = 1
        modemap = []
        if basis.dimension == 1
            # 1D
            modemap = modemap1d
        elseif basis.dimension == 2
            # 2D
            modemap = [
                [
                    i + (j - 1)*(basis.numbernodes1d - 1) for i in modemap1d,
                    j in modemap1d
                ]...,
            ]
        elseif basis.dimension == 3
            # 3D
            modemap = [
                [
                    i +
                    (j - 1)*(basis.numbernodes1d - 1) +
                    (k - 1)*(basis.numbernodes1d - 1)^2 for i in modemap1d,
                    j in modemap1d, k in modemap1d
                ]...,
            ]
        else
            throw(DomanError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
        end
        basis.modemap = modemap
        basis.numbermodes = max(modemap...)
    end

    # return
    return getfield(basis, :modemap)
end

# ------------------------------------------------------------------------------
# get/set property
# ------------------------------------------------------------------------------

function Base.getproperty(basis::TensorBasis, f::Symbol)
    if f == :numbernodes
        return getnumbernodes(basis)
    elseif f == :numberquadraturepoints
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
    elseif f == :numbermodes
        return getnumbermodes(basis)
    elseif f == :modemap
        return getmodemap(basis)
    else
        return getfield(basis, f)
    end
end

function Base.setproperty!(basis::TensorBasis, f::Symbol, value)
    if f == :numbernodes1d ||
       f == :numberquadraturepoints1d ||
       f == :dimension ||
       f == :nodes1d ||
       f == :quadraturepoints1d ||
       f == :quadratureweights1d ||
       f == :interpolation1d ||
       f == :gradient1d ||
       f == :volume
        throw(ReadOnlyMemoryError()) # COV_EXCL_LINE
    else
        return setfield!(basis, f, value)
    end
end

function Base.setproperty!(basis::NonTensorBasis, f::Symbol, value)
    if f == :numbernodes ||
       f == :numberquadraturepoints ||
       f == :dimension ||
       f == :nodes ||
       f == :quadraturepoints ||
       f == :quadratureweights ||
       f == :interpolation ||
       f == :gradient ||
       f == :volume ||
       f == :numbermodes ||
       f == :modemap
        throw(ReadOnlyMemoryError()) # COV_EXCL_LINE
    else
        return setfield!(basis, f, value)
    end
end

# ------------------------------------------------------------------------------
# gradient on stretched mesh
# ------------------------------------------------------------------------------

"""
```julia
getdXdxgradient(basis, mesh)
```

Get gradient adjusted for mesh stretching

# Arguments:
- `basis`: basis to compute gradient
- `mesh`:  mesh to compute gradient

# Returns:
- gradient matrix multiplied by change of coordinates adjoint

# Example:
```jldoctest
for dimension in 1:3
    # mesh
    mesh = []
    if dimension == 1
        mesh = Mesh1D(2.0)
    elseif dimension == 2
        mesh = Mesh2D(2.0, 3.0)
    elseif dimension == 3
        mesh = Mesh3D(2.0, 3.0, 4.0)
    end

    # basis
    basis = TensorH1LagrangeBasis(4, 3, dimension);

    # get gradient on mesh
    gradient = LFAToolkit.getdXdxgradient(basis, mesh);

    # verify
    nodes = basis.nodes;
    linearfunction = [];
    truegradient = [];
    if dimension == 1
        linearfunction = nodes/2;
        truegradient = [1/2*ones(basis.numberquadraturepoints)...]
    elseif dimension == 2
        linearfunction = (nodes[:, 1] + nodes[:, 2])/2;
        truegradient = [
            1/2*ones(basis.numberquadraturepoints)...
            1/3*ones(basis.numberquadraturepoints)...
        ]
    elseif dimension == 3
        linearfunction = (nodes[:, 1] + nodes[:, 2] + nodes[:, 3])/2;
        truegradient = [
            1/2*ones(basis.numberquadraturepoints)...
            1/3*ones(basis.numberquadraturepoints)...
            1/4*ones(basis.numberquadraturepoints)...
        ]
    end

    @assert gradient*linearfunction ≈ truegradient
end

# output

```
"""
function getdXdxgradient(basis::TensorBasis, mesh::Mesh)
    # check compatibility
    dimension = basis.dimension
    if (dimension == 1 && typeof(mesh) != Mesh1D) ||
       (dimension == 2 && typeof(mesh) != Mesh2D) ||
       (dimension == 3 && typeof(mesh) != Mesh3D)
        error("mesh dimension must match basis dimension") # COV_EXCL_LINE
    end

    # get gradient
    gradient = basis.gradient

    # length of reference element
    lengthreference = (max(basis.nodes1d...) - min(basis.nodes1d...))

    # adjust for mesh
    if dimension == 1
        # 1D
        return gradient*lengthreference/mesh.dx
    elseif dimension == 2
        # 2D
        scalex = lengthreference/mesh.dx
        scaley = lengthreference/mesh.dy
        numberquadraturepoints = basis.numberquadraturepoints
        return [
            gradient[1:numberquadraturepoints, :]*scalex
            gradient[numberquadraturepoints+1:end, :]*scaley
        ]
    elseif dimension == 3
        # 3D
        scalex = lengthreference/mesh.dx
        scaley = lengthreference/mesh.dy
        scalez = lengthreference/mesh.dz
        numberquadraturepoints = basis.numberquadraturepoints
        return [
            gradient[1:numberquadraturepoints, :]*scalex
            gradient[numberquadraturepoints+1:2*numberquadraturepoints, :]*scaley
            gradient[2*numberquadraturepoints+1:end, :]*scalez
        ]
    end
end

function getdXdxgradient(basis::NonTensorBasis, mesh::Mesh)
    error("unimplemented")
end

# ------------------------------------------------------------------------------
