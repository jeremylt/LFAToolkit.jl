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
    gradient1d;
    numberelements1d = 1,
)
```

Tensor product basis

# Arguments:

  - `numbernodes1d`:             number of nodes in 1 dimension
  - `numberquadraturepoints1d`:  number of quadrature points in 1 dimension
  - `numbercomponents`:          number of components
  - `dimension`:                 dimension of the basis
  - `nodes1d`:                   coordinates of the nodes in 1 dimension
  - `quadraturepoints1d`:        coordinates of the quadrature points in 1 dimension
  - `quadratureweights1d`:       quadrature weights in 1 dimension
  - `interpolation1d`:           interpolation matrix from nodes to quadrature points in 1 dimension
  - `gradient1d`:                gradient matrix from nodes to quadrature points in 1 dimension

# Keyword Arguments:

  - `numberelements1d = 1`:      number of subelements, for macroelement bases

# Returns:

  - tensor product basis object
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
    primalnodes::AbstractArray{Int,1}
    interfacenodes::AbstractArray{Int,1}

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
            )
            # COV_EXCL_STOP
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
            (maximum(nodes1d) - minimum(nodes1d))^dimension,
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
    gradient;
    numberelements = 1,
)
```

Non-tensor basis

# Arguments:

  - `numbernodes`:             number of nodes
  - `numberquadraturepoints`:  number of quadrature points
  - `numbercomponents`:        number of components
  - `dimension`:               dimension of the basis
  - `nodes`:                   coordinates of the nodes
  - `quadraturepoints`:        coordinates of the quadrature points
  - `quadratureweights`:       quadrature weights
  - `interpolation`:           interpolation matrix from nodes to quadrature points
  - `gradient`:                gradient matrix from nodes to quadrature points

# Keyword Arguments:

  - `numberelements = 1`:      number of subelements, for macroelement bases

# Returns:

  - non-tensor product basis object
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
    primalnodes::AbstractArray{Int,1}
    interfacenodes::AbstractArray{Int,1}

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
            volume *= (maximum(nodes[:, d]) - minimum(nodes[:, d]))
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
            maximum(modemap),
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

  - `basis`:  basis to compute number of nodes

# Returns:

  - integer number of basis nodes

# Example:

```jldoctest
# setup basis
basis = TensorH1LagrangeBasis(4, 3, 2, 2);

# verify number of nodes
@assert basis.numbernodes == 4^2

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

# Arguments:

  - `basis`:  basis to compute nodes

# Returns:

  - basis nodes array

# Example:

```jldoctest
# test for all supported dimensions
for dimension = 1:3
    # setup basis
    basis = TensorH1LagrangeBasis(4, 3, 2, dimension)

    # get true nodes
    truenodes1d = [-1, -√(1 / 5), √(1 / 5), 1]
    truenodes = []
    if dimension == 1
        truenodes = truenodes1d
    elseif dimension == 2
        truenodes = transpose(hcat([[[x, y] for x in truenodes1d, y in truenodes1d]...]...))
    elseif dimension == 3
        truenodes = transpose(
            hcat(
                [
                    [[x, y, z] for x in truenodes1d, y in truenodes1d, z in truenodes1d]...,
                ]...,
            ),
        )
    end

    # verify basis nodes
    @assert basis.nodes ≈ truenodes
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
            throw(DomainError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
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

  - `basis`:  basis to compute number of quadrature points

# Returns:

  - integer number of basis quadrature points

# Example:

```jldoctest
# setup basis
basis = TensorH1LagrangeBasis(4, 3, 2, 2);

# verify number of quadrature points
@assert basis.numberquadraturepoints == 3^2

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

# Arguments:

  - `basis`: basis to compute quadrature points

# Returns:

  - basis quadrature points array

# Example:

```jldoctest
# test for all supported dimensions
for dimension = 1:3
    # setup basis
    basis = TensorH1LagrangeBasis(4, 3, 2, dimension)

    # get true quadrature points
    truequadraturepoints1d = [-√(3 / 5), 0, √(3 / 5)]
    truequadraturepoints = []
    if dimension == 1
        truequadraturepoints = truequadraturepoints1d
    elseif dimension == 2
        truequadraturepoints = transpose(
            hcat(
                [
                    [
                        [x, y] for x in truequadraturepoints1d, y in truequadraturepoints1d
                    ]...,
                ]...,
            ),
        )
    elseif dimension == 3
        truequadraturepoints = transpose(
            hcat(
                [
                    [
                        [x, y, z] for x in truequadraturepoints1d,
                        y in truequadraturepoints1d, z in truequadraturepoints1d
                    ]...,
                ]...,
            ),
        )
    end

    # verify quadrature points
    @assert basis.quadraturepoints ≈ truequadraturepoints
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
            throw(DomainError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
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

# Arguments:

  - `basis`:  basis to compute quadrature weights

# Returns:

  - basis quadrature weights vector

# Example:

```jldoctest
# test for all supported dimensions
for dimension = 1:3
    # setup basis
    basis = TensorH1LagrangeBasis(4, 3, 2, dimension)

    # get true quadratrue weights
    trueweights1d = [5 / 9, 8 / 9, 5 / 9]
    trueweights = []
    if dimension == 1
        trueweights = trueweights1d
    elseif dimension == 2
        trueweights = kron(trueweights1d, trueweights1d)
    elseif dimension == 3
        trueweights = kron(trueweights1d, trueweights1d, trueweights1d)
    end

    # verify
    @assert basis.quadratureweights ≈ trueweights
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
            throw(DomainError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
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

  - `basis`:  basis to compute interpolation matrix

# Returns:

  - basis interpolation matrix

# Example:

```jldoctest
# test for all supported dimensions
for dimension = 1:3
    # setup basis
    basis = TensorH1LagrangeBasis(4, 3, 2, dimension)

    # get basis interpolation matrix
    interpolation = basis.interpolation

    # verify
    for i = 1:3^dimension
        total = sum(interpolation[i, :])
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
            throw(DomainError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
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

  - `basis`:  basis to compute gradient matrix

# Returns:

  - basis gradient matrix

# Example:

```jldoctest
# test for all supported dimensions
for dimension = 1:3
    # setup basis
    basis = TensorH1LagrangeBasis(4, 3, 2, dimension)

    # get basis gradient matrix
    gradient = basis.gradient

    # verify
    for i = 1:dimension*3^dimension
        total = sum(gradient[i, :])
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
            throw(DomainError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
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

  - `basis`:  basis to compute number of modes

# Returns:

  - number of modes for basis

# Example:

```jldoctest
# test for all supported dimensions
for dimension = 1:3
    # setup basis
    basis = TensorH1LagrangeBasis(4, 3, 2, dimension)

    # verify number of basis modes
    @assert basis.numbermodes == 2 * 3^dimension
end

# output

```
"""
function getnumbermodes(basis::TensorBasis)
    # assemble if needed
    if !isdefined(basis, :modemap)
        basis.numbermodes = maximum(basis.modemap)
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

  - `basis`:  basis to compute mode map vector

# Returns:

  - basis mode map vector

# Example:

```jldoctest
# test for all supported dimensions
for dimension = 1:3
    # setup basis
    basis = TensorH1LagrangeBasis(4, 3, 1, dimension)

    # get true mode map
    truemodemap1d = [1, 2, 3, 1]
    truemodemap = []
    if dimension == 1
        truemodemap = truemodemap1d
    elseif dimension == 2
        truemodemap = [[i + (j - 1) * 3 for i in truemodemap1d, j in truemodemap1d]...]
    elseif dimension == 3
        truemodemap = [
            [
                i + (j - 1) * 3 + (k - 1) * 3^2 for i in truemodemap1d,
                j in truemodemap1d, k in truemodemap1d
            ]...,
        ]
    end

    # verify
    @assert basis.modemap == truemodemap
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
            throw(DomainError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
        end
        numbermodes1component = maximum(modemap)
        modemap = vcat(
            [
                ((i - 1) * numbermodes1component) .+ modemap for
                i = 1:basis.numbercomponents
            ]...,
        )
        basis.modemap = modemap
        basis.numbermodes = maximum(modemap)
    end

    # return
    return getfield(basis, :modemap)
end

"""
```julia
getprimalnodes(basis)
```

Get primal nodes for basis

# Arguments:

  - `basis`:  basis to compute primal nodes

# Returns:

  - basis primal nodes vector

# Example:

```jldoctest
# test for all supported dimensions
for dimension = 1:3
    # setup basis
    p = 4
    basis = TensorH1LagrangeBasis(p, p, 1, dimension)

    # get true primal modes
    trueprimalnodes = []
    if dimension == 1
        trueprimalnodes = [1, p]
    elseif dimension == 2
        trueprimalnodes = [1, p, p^2 - p + 1, p^2]
    elseif dimension == 3
        trueprimalnodes =
            [1, p, p^2 - p + 1, p^2, p^3 - p^2 + 1, p^3 - p^2 + p, p^3 - p + 1, p^3]
    end

    # verify
    @assert basis.primalnodes == trueprimalnodes
end

# output

```
"""
function getprimalnodes(basis::TensorBasis)
    # assemble if needed
    if !isdefined(basis, :primalnodes)
        primalnodes1component = []
        p = basis.numbernodes1d
        if basis.dimension == 1
            # 1D
            primalnodes1component = [1, p]
        elseif basis.dimension == 2
            # 2D
            primalnodes1component = [1, p, p^2 - p + 1, p^2]
        elseif basis.dimension == 3
            # 3D
            primalnodes1component =
                [1, p, p^2 - p + 1, p^2, p^3 - p^2 + 1, p^3 - p^2 + p, p^3 - p + 1, p^3]
        else
            throw(DomainError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
        end
        primalnodes = vcat(
            [
                (i - 1) * basis.numbernodes .+ primalnodes1component for
                i = 1:basis.numbercomponents
            ]...,
        )
        basis.primalnodes = primalnodes
    end

    # return
    return getfield(basis, :primalnodes)
end

"""
```julia
getinterfacenodes(basis)
```

Get interface nodes for basis

# Arguments:

  - `basis`:  basis to compute primal nodes

# Returns:

  - basis primal nodes vector

# Example:

```jldoctest
# test for all supported dimensions
for dimension = 1:3
    # setup basis
    p = 4
    basis = TensorH1LagrangeBasis(p, p, 1, dimension)

    # get true interface nodes
    trueinterfacenodes = []
    if dimension == 1
        trueinterfacenodes = [1, p]
    elseif dimension == 2
        trueinterfacenodes = [1:p..., p^2-p+1:p^2...]
        for i = 1:p-2
            push!(trueinterfacenodes, i * p + 1)
            push!(trueinterfacenodes, (i + 1) * p)
        end
    elseif dimension == 3
        trueinterfacenodes = [1:p^2..., p^3-p^2+1:p^3...]
        for i = 1:p-2
            push!(trueinterfacenodes, i*p^2+1:i*p^2+p...)
            push!(trueinterfacenodes, (i+1)*p^2-p+1:(i+1)*p^2...)
            push!(trueinterfacenodes, i*p^2+p+1:p:(i+1)*p^2-2*p+1...)
            push!(trueinterfacenodes, i*p^2+2*p:p:(i+1)*p^2-p...)
        end
    end
    trueinterfacenodes = sort(trueinterfacenodes)

    # verify
    @assert basis.interfacenodes == trueinterfacenodes
end

# output

```
"""
function getinterfacenodes(basis::TensorBasis)
    # assemble if needed
    if !isdefined(basis, :interfacenodes)
        interfacenodes1component = []
        p = basis.numbernodes1d
        if basis.dimension == 1
            # 1D
            interfacenodes1component = [1, p]
        elseif basis.dimension == 2
            # 2D
            interfacenodes1component = [1:p..., p^2-p+1:p^2...]
            for i = 1:p-2
                push!(interfacenodes1component, i * p + 1)
                push!(interfacenodes1component, (i + 1) * p)
            end
        elseif basis.dimension == 3
            # 3D
            interfacenodes1component = [1:p^2..., p^3-p^2+1:p^3...]
            for i = 1:p-2
                push!(interfacenodes1component, i*p^2+1:i*p^2+p...)
                push!(interfacenodes1component, (i+1)*p^2-p+1:(i+1)*p^2...)
                push!(interfacenodes1component, i*p^2+p+1:p:(i+1)*p^2-2*p+1...)
                push!(interfacenodes1component, i*p^2+2*p:p:(i+1)*p^2-p...)
            end
        else
            throw(DomainError(basis.dimension, "Dimension must be less than or equal to 3")) # COV_EXCL_LINE
        end
        interfacenodes1component = sort(interfacenodes1component)
        interfacenodes = vcat(
            [
                (i - 1) * basis.numbernodes .+ interfacenodes1component for
                i = 1:basis.numbercomponents
            ]...,
        )
        basis.interfacenodes = interfacenodes
    end

    # return
    return getfield(basis, :interfacenodes)
end

"""
```julia
getnumberelements(basis)
```

Get the number of elements for the basis

# Arguments:

  - `basis`:  basis to compute number of micro-elements

# Returns:

  - integer number of basis micro-elements

# Example:

```jldoctest
# get number of nodes for basis
basis = TensorH1LagrangeMacroBasis(4, 4, 1, 2, 2);

# verify number of elements
@assert basis.numberelements == 2^2

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
    elseif f == :primalnodes
        return getprimalnodes(basis)
    elseif f == :interfacenodes
        return getinterfacenodes(basis)
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

  - `basis`:  basis to compute gradient
  - `mesh`:   mesh to compute gradient

# Returns:

  - gradient matrix multiplied by change of coordinates adjoint

# Example:

```jldoctest
for dimension = 1:3
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
    basis = TensorH1LagrangeBasis(4, 3, 1, dimension)

    # get gradient on mesh
    gradient = LFAToolkit.getdXdxgradient(basis, mesh)

    # verify
    nodes = basis.nodes
    linearfunction = []
    truegradient = []
    if dimension == 1
        linearfunction = nodes
        truegradient = [1 / 2 * ones(basis.numberquadraturepoints)...]
    elseif dimension == 2
        linearfunction = (nodes[:, 1] + nodes[:, 2])
        truegradient = [
            1 / 2 * ones(basis.numberquadraturepoints)...
            1 / 3 * ones(basis.numberquadraturepoints)...
        ]
    elseif dimension == 3
        linearfunction = (nodes[:, 1] + nodes[:, 2] + nodes[:, 3])
        truegradient = [
            1 / 2 * ones(basis.numberquadraturepoints)...
            1 / 3 * ones(basis.numberquadraturepoints)...
            1 / 4 * ones(basis.numberquadraturepoints)...
        ]
    end

    @assert gradient * linearfunction ≈ truegradient
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

    # coordinate transformation
    gradient = basis.gradient

    # adjust for mesh
    if dimension == 1
        # 1D
        return gradient / mesh.dx
    elseif dimension == 2
        # 2D
        numberquadraturepoints = basis.numberquadraturepoints
        return [
            gradient[1:numberquadraturepoints, :] / mesh.dx
            gradient[numberquadraturepoints+1:end, :] / mesh.dy
        ]
    elseif dimension == 3
        # 3D
        numberquadraturepoints = basis.numberquadraturepoints
        return [
            gradient[1:numberquadraturepoints, :] / mesh.dx
            gradient[numberquadraturepoints+1:2*numberquadraturepoints, :] / mesh.dy
            gradient[2*numberquadraturepoints+1:end, :] / mesh.dz
        ]
    end
end

function getdXdxgradient(basis::NonTensorBasis, mesh::Mesh)
    throw(error("dXdxgradient unimplemented for non-tensor bases"))
end

"""
```julia
getdxdXquadratureweights(basis, mesh)
```

Get quadrature weights adjusted for mesh stretching

# Arguments:

  - `basis`:  basis to compute quadratureweights
  - `mesh`:   mesh to compute quadratureweights

# Returns:

  - quadrature weights multiplied by change of coordinates adjoint

# Example:

```jldoctest
mesh = Mesh1D(2.0)

# basis
basis = TensorH1LagrangeBasis(4, 3, 1, 1);

# get gradient on mesh
weights = LFAToolkit.getdxdXquadratureweights(basis, mesh);

# verify
@assert basis.quadratureweights * mesh.volume / basis.volume ≈ weights

# output

```
"""
function getdxdXquadratureweights(basis::TensorBasis, mesh::Mesh)
    # check compatibility
    dimension = basis.dimension
    if (dimension == 1 && typeof(mesh) != Mesh1D) ||
       (dimension == 2 && typeof(mesh) != Mesh2D) ||
       (dimension == 3 && typeof(mesh) != Mesh3D)
        error("mesh dimension must match basis dimension") # COV_EXCL_LINE
    end

    # adjust for mesh
    return basis.quadratureweights * mesh.volume / basis.volume
end

function getdxdXquadratureweights(basis::NonTensorBasis, mesh::Mesh)
    throw(error("dxdXquadratureweights unimplemented for non-tensor bases"))
end

# ------------------------------------------------------------------------------
