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
    numbercomponents,
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
- `numbercomponents`:         number of components
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
    numbercomponents::Int
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
    primalvertices::AbstractArray{Int,1}
    interfacevertices::AbstractArray{Int,1}

    # inner constructor
    TensorBasis(
        numbernodes1d::Int,
        numberquadraturepoints1d::Int,
        numbercomponents::Int,
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
            error("number of basis nodes in 1d must be at least 1") # COV_EXCL_LINE
        end;
        if numberquadraturepoints1d < 1
            error("number of quadrature points in 1d must be at least 1") # COV_EXCL_LINE
        end;
        if numbercomponents < 1
            error("number of components must be at least 1") # COV_EXCL_LINE
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
            numbercomponents,
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
        basis.numberelements1d == 1 ? "" : "macro-element ",
        "tensor product basis:\n    numbernodes1d: ",
        basis.numbernodes1d,
        "\n    numberquadraturepoints1d: ",
        basis.numberquadratuepoints1d,
        "\n    numbercomponents: ",
        basis.numbercomponents,
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
    numbercomponents,
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
- `numbercomponents`:       number of components
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
    numbercomponents::Int
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
    primalvertices::AbstractArray{Int,1}
    interfacevertices::AbstractArray{Int,1}

    # inner constructor
    NonTensorBasis(
        numbernodes::Int,
        numberquadraturepoints::Int,
        numbercomponents::Int,
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
        if numbercomponents < 1
            error("number of components must be at least 1") # COV_EXCL_LINE
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
        if size(gradient) != (q * dimension, numbernodes)
            # COV_EXCL_START
            error(
                "gradient matrix must have dimensions (numberquadraturepoints*dimension, numbernodes)",
            )
            # COV_EXCL_STOP
        end;
        if length(modemap) != numbercomponents * numbernodes
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
            numbercomponents,
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
        basis.numberelements == 1 ? "" : "macro-element ",
        "non-tensor product basis:\n    numbernodes: ",
        basis.numbernodes,
        "\n    numberquadraturepoints: ",
        basis.numberquadratuepoints,
        "\n    numbercomponents: ",
        basis.numbercomponents,
    )
    if basis.numberelements != 1
        print(io, "\n    numberelements: ", basis.numberelements)
    end
    print(io, "\n    dimension: ", basis.dimension)
end
# COV_EXCL_STOP

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
basis = TensorH1LagrangeBasis(4, 3, 2, 2);

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
    basis = TensorH1LagrangeBasis(4, 3, 2, dimension);

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
basis = TensorH1LagrangeBasis(4, 3, 2, 2);

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
    basis = TensorH1LagrangeBasis(4, 3, 2, dimension);

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
    basis = TensorH1LagrangeBasis(4, 3, 2, dimension);

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
    basis = TensorH1LagrangeBasis(4, 3, 2, dimension);

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
            interpolation = kron(I(basis.numbercomponents), basis.interpolation1d)
        elseif basis.dimension == 2
            # 2D
            interpolation = kron(
                I(basis.numbercomponents),
                basis.interpolation1d,
                basis.interpolation1d,
            )
        elseif basis.dimension == 3
            # 3D
            interpolation = kron(
                I(basis.numbercomponents),
                basis.interpolation1d,
                basis.interpolation1d,
                basis.interpolation1d,
            )
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
    basis = TensorH1LagrangeBasis(4, 3, 2, dimension);

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
            gradient = kron(I(basis.numbercomponents), basis.gradient1d)
        elseif basis.dimension == 2
            # 2D
            gradient = kron(
                I(basis.numbercomponents),
                [
                    kron(basis.gradient1d, basis.interpolation1d)
                    kron(basis.interpolation1d, basis.gradient1d)
                ],
            )
        elseif basis.dimension == 3
            # 3D
            gradient = kron(
                I(basis.numbercomponents),
                [
                    kron(basis.gradient1d, basis.interpolation1d, basis.interpolation1d)
                    kron(basis.interpolation1d, basis.gradient1d, basis.interpolation1d)
                    kron(basis.interpolation1d, basis.interpolation1d, basis.gradient1d)
                ],
            )
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
    basis = TensorH1LagrangeBasis(4, 3, 2, dimension);

    # note: either syntax works
    numbermodes = LFAToolkit.getnumbermodes(basis);
    numbermodes = basis.numbermodes;

    # verify
    @assert numbermodes == 2*3^dimension
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
    basis = TensorH1LagrangeBasis(4, 3, 1, dimension);

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
                    i + (j - 1) * (basis.numbernodes1d - 1) for i in modemap1d,
                    j in modemap1d
                ]...,
            ]
        elseif basis.dimension == 3
            # 3D
            modemap = [
                [
                    i +
                    (j - 1) * (basis.numbernodes1d - 1) +
                    (k - 1) * (basis.numbernodes1d - 1)^2 for i in modemap1d,
                    j in modemap1d, k in modemap1d
                ]...,
            ]
        else
            throw(DomanError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
        end
        numbermodes1component = max(modemap...)
        modemap = vcat(
            [
                ((i - 1) * numbermodes1component) .+ modemap for
                i = 1:basis.numbercomponents
            ]...,
        )
        basis.modemap = modemap
        basis.numbermodes = max(modemap...)
    end

    # return
    return getfield(basis, :modemap)
end

"""
```julia
getprimalvertices(basis)
```

Get primal vertices for basis

# Arguments:
- `basis`: basis to compute primal vertices

# Returns:
- Basis primal vertices vector

# Example:
```jldoctest
# test for all supported dimensions
for dimension in 1:3
    # get mode map vector
    p = 4
    basis = TensorH1LagrangeBasis(p, p, 1, dimension);

    # note: either syntax works
    primalvertices = LFAToolkit.getprimalvertices(basis);
    primalvertices = basis.primalvertices;

    # verify
    trueprimalvertices = []
    if dimension == 1
        trueprimalvertices = [1, p];
    elseif dimension == 2
        trueprimalvertices = [1, p, p^2-p+1, p^2]
    elseif dimension == 3
        trueprimalvertices = [1, p, p^2-p+1, p^2, p^3-p^2+1, p^3-p^2+p, p^3-p+1, p^3]
    end

    @assert trueprimalvertices == primalvertices
end

# output

```
"""
function getprimalvertices(basis::TensorBasis)
    # assemble if needed
    if !isdefined(basis, :primalvertices)
        primalvertices1component = []
        p = basis.numbernodes1d
        if basis.dimension == 1
            # 1D
            primalvertices1component = [1, p]
        elseif basis.dimension == 2
            # 2D
            primalvertices1component = [1, p, p^2 - p + 1, p^2]
        elseif basis.dimension == 3
            # 3D
            primalvertices1component =
                [1, p, p^2 - p + 1, p^2, p^3 - p^2 + 1, p^3 - p^2 + p, p^3 - p + 1, p^3]
        else
            throw(DomanError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
        end
        primalvertices = vcat(
            [
                (i - 1) * basis.numbernodes .+ primalvertices1component for
                i = 1:basis.numbercomponents
            ]...,
        )
        basis.primalvertices = primalvertices
    end

    # return
    return getfield(basis, :primalvertices)
end

"""
```julia
getinterfacevertices(basis)
```

Get interface vertices for basis

# Arguments:
- `basis`: basis to compute primal vertices

# Returns:
- Basis primal vertices vector

# Example:
```jldoctest
# test for all supported dimensions
for dimension in 1:3
    # get mode map vector
    p = 4
    basis = TensorH1LagrangeBasis(p, p, 1, dimension);

    # note: either syntax works
    interfacevertices = LFAToolkit.getinterfacevertices(basis);
    interfacevertices = basis.interfacevertices;

    # verify
    trueinterfacevertices = []
    if dimension == 1
        trueinterfacevertices = [1, p];
    elseif dimension == 2
        trueinterfacevertices = [1:p..., p^2-p+1:p^2...]
        for i = 1:p-2
            push!(trueinterfacevertices, i*p+1)
            push!(trueinterfacevertices, (i+1)*p)
        end
    elseif dimension == 3
        trueinterfacevertices = [1:p^2..., p^3-p^2+1:p^3...]
        for i = 1:p-2
            push!(trueinterfacevertices, i*p^2+1:i*p^2+p...)
            push!(trueinterfacevertices, (i+1)*p^2-p+1:(i+1)*p^2...)
            push!(trueinterfacevertices, i*p^2+p+1:p:(i+1)*p^2-2*p+1...)
            push!(trueinterfacevertices, i*p^2+2*p:p:(i+1)*p^2-p...)
        end
    end
    trueinterfacevertices = sort(trueinterfacevertices)

    @assert trueinterfacevertices == interfacevertices
end

# output

```
"""
function getinterfacevertices(basis::TensorBasis)
    # assemble if needed
    if !isdefined(basis, :interfacevertices)
        interfacevertices1component = []
        p = basis.numbernodes1d
        if basis.dimension == 1
            # 1D
            interfacevertices1component = [1, p]
        elseif basis.dimension == 2
            # 2D
            interfacevertices1component = [1:p..., p^2-p+1:p^2...]
            for i = 1:p-2
                push!(interfacevertices1component, i * p + 1)
                push!(interfacevertices1component, (i + 1) * p)
            end
        elseif basis.dimension == 3
            # 3D
            interfacevertices1component = [1:p^2..., p^3-p^2+1:p^3...]
            for i = 1:p-2
                push!(interfacevertices1component, i*p^2+1:i*p^2+p...)
                push!(interfacevertices1component, (i+1)*p^2-p+1:(i+1)*p^2...)
                push!(interfacevertices1component, i*p^2+p+1:p:(i+1)*p^2-2*p+1...)
                push!(interfacevertices1component, i*p^2+2*p:p:(i+1)*p^2-p...)
            end
        else
            throw(DomanError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
        end
        interfacevertices1component = sort(interfacevertices1component)
        interfacevertices = vcat(
            [
                (i - 1) * basis.numbernodes .+ interfacevertices1component for
                i = 1:basis.numbercomponents
            ]...,
        )
        basis.interfacevertices = interfacevertices
    end

    # return
    return getfield(basis, :interfacevertices)
end

"""
```julia
getnumberelements(basis)
```

Get the number of elements for the basis

# Arguments:
- `basis`: basis to compute number of micro-elements

# Returns:
- Integer number of basis micro-elements

# Example:
```jldoctest
# get number of nodes for basis
basis = TensorH1LagrangeMacroBasis(4, 4, 1, 2, 2);

# note: either syntax works
numbernodes = LFAToolkit.getnumberelements(basis);
numbernodes = basis.numberelements;

# verify
@assert numbernodes == 2^2

# output

```
"""
function getnumberelements(basis::TensorBasis)
    return basis.numberelements1d^basis.dimension
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
    elseif f == :primalvertices
        return getprimalvertices(basis)
    elseif f == :interfacevertices
        return getinterfacevertices(basis)
    elseif f == :numberelements
        return getnumberelements(basis)
    else
        return getfield(basis, f)
    end
end

function Base.setproperty!(basis::TensorBasis, f::Symbol, value)
    if f == :numbernodes1d ||
       f == :numberquadraturepoints1d ||
       f == :numbercomponents ||
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
       f == :numbercomponents ||
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
    basis = TensorH1LagrangeBasis(4, 3, 1, dimension);

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
        return gradient * lengthreference / mesh.dx
    elseif dimension == 2
        # 2D
        scalex = lengthreference / mesh.dx
        scaley = lengthreference / mesh.dy
        numberquadraturepoints = basis.numberquadraturepoints
        return [
            gradient[1:numberquadraturepoints, :] * scalex
            gradient[numberquadraturepoints+1:end, :] * scaley
        ]
    elseif dimension == 3
        # 3D
        scalex = lengthreference / mesh.dx
        scaley = lengthreference / mesh.dy
        scalez = lengthreference / mesh.dz
        numberquadraturepoints = basis.numberquadraturepoints
        return [
            gradient[1:numberquadraturepoints, :] * scalex
            gradient[numberquadraturepoints+1:2*numberquadraturepoints, :] * scaley
            gradient[2*numberquadraturepoints+1:end, :] * scalez
        ]
    end
end

function getdXdxgradient(basis::NonTensorBasis, mesh::Mesh)
    throw(error("dXdxgradient unimplemented for non-tensor bases"))
end

# ------------------------------------------------------------------------------
