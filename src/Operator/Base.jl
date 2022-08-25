# ------------------------------------------------------------------------------
# finite element operators
# ------------------------------------------------------------------------------

"""
```julia
Operator(
    weakform,
    mesh,
    inputs,
    outputs
)
```

Finite element operator comprising of a weak form and bases

# Arguments:
- `weakform`:  user provided function that represents weak form at
                   quadrature points
- `mesh`:      mesh object with deformation in each dimension
- `inputs`:    array of operator input fields
- `outputs`:   array of operator output fields

# Returns:
- Finite element operator object

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
basis = TensorH1LagrangeBasis(4, 4, 1, 2);

function massweakform(u::Array{Float64}, w::Array{Float64})
    v = u*w[1]
    return [v]
end

# mass operator
inputs = [
    OperatorField(basis, [EvaluationMode.interpolation]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.interpolation])];
mass = Operator(massweakform, mesh, inputs, outputs);

# verify
println(mass)

# output
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
mutable struct Operator
    # data never changed
    weakform::Function
    mesh::Mesh
    inputs::AbstractArray{OperatorField}
    outputs::AbstractArray{OperatorField}

    # data empty until assembled
    elementmatrix::AbstractArray{Float64,2}
    diagonal::AbstractArray{Float64}
    multiplicity::AbstractArray{Float64}
    rowmodemap::AbstractArray{Float64,2}
    columnmodemap::AbstractArray{Float64,2}
    inputcoordinates::AbstractArray{Float64}
    outputcoordinates::AbstractArray{Float64}
    nodecoordinatedifferences::AbstractArray{Float64}

    # inner constructor
    Operator(
        weakform,
        mesh::Mesh,
        inputs::AbstractArray{OperatorField},
        outputs::AbstractArray{OperatorField},
    ) = (
        dimension = 0;
        numberquadraturepoints = 0;

        # check inputs valididy
        if length(inputs) < 1
            error("must have at least one input") # COV_EXCL_LINE
        end;
        for input in inputs
            # dimension
            if dimension == 0
                dimension = input.basis.dimension
            end
            if input.basis.dimension != dimension
                error("bases must have compatible dimensions") # COV_EXCL_LINE
            end

            # number of quadrature points
            if numberquadraturepoints == 0
                numberquadraturepoints = input.basis.numberquadraturepoints
            end
            if input.basis.numberquadraturepoints != numberquadraturepoints
                error("bases must have compatible quadrature spaces") # COV_EXCL_LINE
            end
        end;

        # check outputs valididy
        if length(outputs) < 1
            error("must have at least one output") # COV_EXCL_LINE
        end;
        for output in outputs
            # evaluation modes
            if EvaluationMode.quadratureweights in output.evaluationmodes
                error("quadrature weights is not a valid output") # COV_EXCL_LINE
            end

            # dimension
            if output.basis.dimension != dimension
                error("bases must have compatible dimensions") # COV_EXCL_LINE
            end

            # number of quadrature points
            if output.basis.numberquadraturepoints != numberquadraturepoints
                error("bases must have compatible quadrature spaces") # COV_EXCL_LINE
            end
        end;

        # check mesh valididy
        if (dimension == 1 && typeof(mesh) != Mesh1D) ||
           (dimension == 2 && typeof(mesh) != Mesh2D) ||
           (dimension == 3 && typeof(mesh) != Mesh3D)
            error("mesh dimension must match bases dimension") # COV_EXCL_LINE
        end;

        # constructor
        new(weakform, mesh, inputs, outputs)
    )
end

# printing
# COV_EXCL_START
function Base.show(io::IO, operator::Operator)
    print(io, "finite element operator:\n", operator.mesh)

    # inputs
    if length(operator.inputs) == 1
        print(io, "\n\n1 input:")
    else
        print(io, "\n\n", length(operator.inputs), " inputs:")
    end
    for i = 1:length(operator.inputs)
        print(io, "\n", operator.inputs[i])
    end

    # outputs
    if length(operator.outputs) == 1
        print(io, "\n\n1 output:")
    else
        print("\n\n", length(operator.outputs), " outputs:")
    end
    for i = 1:length(operator.outputs)
        print(io, "\n", operator.outputs[i])
    end
end
# COV_EXCL_STOP

# ------------------------------------------------------------------------------
# data for computing symbols
# ------------------------------------------------------------------------------

"""
```julia
getdimension(operator)
```

Retrieve the dimension of an operator

# Arguments:
- `operator`:  operator to retrieve the dimension

# Returns:
- Dimension of the operator

# Example
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
mass = GalleryOperator("mass", 4, 4, mesh);

# verify
@assert mass.dimension == 2

# output

```
"""
function getdimension(operator::Operator)
    return operator.mesh.dimension
end


"""
```julia
getelementmatrix(operator)
```

Compute or retrieve the element matrix of operator for computing the symbol

# Arguments:
- `operator`:  operator to compute element matrix

# Returns:
- Assembled element matrix

# Mass matrix example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
mass = GalleryOperator("mass", 4, 4, mesh);

# element matrix computation
# note: either syntax works
elementmatrix = LFAToolkit.getelementmatrix(mass);
elementmatrix = mass.elementmatrix;

# verify
u = ones(4*4);
v = elementmatrix*u;

total = sum(v);
@assert total ≈ 1.0

# output

```

# Diffusion matrix example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
diffusion = GalleryOperator("diffusion", 4, 4, mesh);

# element matrix computation
# note: either syntax works
elementmatrix = LFAToolkit.getelementmatrix(diffusion);
elementmatrix = diffusion.elementmatrix;

# verify
u = ones(4*4);
v = elementmatrix*u;

total = sum(v);
@assert abs(total) < 1e-14

# output

```
"""
function getelementmatrix(operator::Operator)
    # assemble if needed
    if !isdefined(operator, :elementmatrix)
        # collect info on operator
        weakforminputs = []
        numbernodes = 0
        numbernodeinputs = 0
        numbernodeoutputs = 0
        numberquadraturepoints = 0
        numberquadratureinputs = 0
        numberquadratureoutputs = 0
        numberfieldsin = []
        numberfieldsout = []
        weightinputindex = 0
        quadratureweights = []
        Bblocks = []
        Btblocks = []

        # inputs
        for input in operator.inputs
            # number of nodes
            if input.evaluationmodes[1] != EvaluationMode.quadratureweights
                numbernodes += input.basis.numbernodes
            end

            # number of quadrature points
            if numberquadraturepoints == 0
                numberquadraturepoints = input.basis.numberquadraturepoints
            end

            # input evaluation modes
            if input.evaluationmodes[1] == EvaluationMode.quadratureweights
                push!(weakforminputs, zeros(1))
                weightinputindex = findfirst(isequal(input), operator.inputs)
            else
                numberfields = 0
                Bcurrent = []
                for mode in input.evaluationmodes
                    if mode == EvaluationMode.interpolation
                        numberfields += 1
                        numbernodeinputs += input.basis.numbercomponents
                        numberquadratureinputs += input.basis.numbercomponents
                        Bcurrent =
                            Bcurrent == [] ? input.basis.interpolation :
                            [Bcurrent; input.basis.interpolation]
                    elseif mode == EvaluationMode.gradient
                        numberfields += input.basis.dimension
                        numbernodeinputs += input.basis.numbercomponents
                        numberquadratureinputs +=
                            input.basis.dimension * input.basis.numbercomponents
                        gradient = getdXdxgradient(input.basis, operator.mesh)
                        Bcurrent = Bcurrent == [] ? gradient : [Bcurrent; gradient]
                    end
                end
                push!(Bblocks, Bcurrent)
                push!(weakforminputs, zeros(numberfields, input.basis.numbercomponents))
                push!(numberfieldsin, numberfields * input.basis.numbercomponents)
            end
        end

        # input basis matrix
        B = spzeros(
            numberquadratureinputs * numberquadraturepoints,
            numbernodeinputs * numbernodes,
        )
        currentrow = 1
        currentcolumn = 1
        for Bblock in Bblocks
            B[
                currentrow:currentrow+size(Bblock)[1]-1,
                currentcolumn:currentcolumn+size(Bblock)[2]-1,
            ] = Bblock
            currentrow += size(Bblock)[1]
            currentcolumn += size(Bblock)[2]
        end

        # quadrature weight input index
        if weightinputindex != 0
            quadratureweights = getdxdXquadratureweights(
                operator.inputs[weightinputindex].basis,
                operator.mesh,
            )
        end

        # outputs
        for output in operator.outputs
            # output evaluation modes
            numberfields = 0
            Btcurrent = []
            for mode in output.evaluationmodes
                if mode == EvaluationMode.interpolation
                    numberfields += output.basis.numbercomponents
                    numbernodeoutputs += output.basis.numbercomponents
                    numberquadratureoutputs += output.basis.numbercomponents
                    Btcurrent =
                        Btcurrent == [] ? output.basis.interpolation :
                        [Btcurrent; output.basis.intepolation]
                elseif mode == EvaluationMode.gradient
                    numberfields += output.basis.dimension * output.basis.numbercomponents
                    numbernodeoutputs += output.basis.numbercomponents
                    numberquadratureoutputs +=
                        output.basis.dimension * output.basis.numbercomponents
                    gradient = getdXdxgradient(output.basis, operator.mesh)
                    Btcurrent = Btcurrent == [] ? gradient : [Btcurrent; gradient]
                    # note: quadrature weights checked in constructor
                end
            end
            push!(Btblocks, Btcurrent)
            push!(numberfieldsout, numberfields)
        end

        # output basis matrix
        Bt = spzeros(
            numberquadratureoutputs * numberquadraturepoints,
            numbernodeoutputs * numbernodes,
        )
        currentrow = 1
        currentcolumn = 1
        for Btblock in Btblocks
            Bt[
                currentrow:currentrow+size(Btblock)[1]-1,
                currentcolumn:currentcolumn+size(Btblock)[2]-1,
            ] = Btblock
            currentrow += size(Btblock)[1]
            currentcolumn += size(Btblock)[2]
        end
        Bt = Bt'

        # QFunction matrix
        D = spzeros(
            numberquadratureoutputs * numberquadraturepoints,
            numberquadratureinputs * numberquadraturepoints,
        )
        # loop over inputs
        currentfieldin = 0
        for i = 1:length(operator.inputs)
            input = operator.inputs[i]
            if input.evaluationmodes[1] == EvaluationMode.quadratureweights
                continue
            end

            # loop over quadrature points
            for q = 1:numberquadraturepoints
                # set quadrature weight
                if weightinputindex != 0
                    weakforminputs[weightinputindex][1] = quadratureweights[q]
                end

                # fill sparse matrix
                for j = 1:numberfieldsin[i]
                    # run user weak form function
                    weakforminputs[i][j] = 1.0
                    outputs = operator.weakform(weakforminputs...)
                    weakforminputs[i][j] = 0.0

                    # store outputs
                    currentfieldout = 0
                    for k = 1:length(operator.outputs)
                        for l = 1:numberfieldsout[k]
                            D[
                                currentfieldout*numberquadraturepoints+q,
                                (currentfieldin+j-1)*numberquadraturepoints+q,
                            ] = outputs[k][l]
                            currentfieldout += 1
                        end
                    end
                end
            end
            currentfieldin += numberfieldsin[i]
        end

        # multiply A = B^T D B and store
        elementmatrix = Bt * D * B
        operator.elementmatrix = elementmatrix
    end

    # return
    return getfield(operator, :elementmatrix)
end

"""
```julia
getdiagonal(operator)
```

Compute or retrieve the symbol matrix diagonal for an operator

# Arguments:
- `operator`:  operator to compute diagonal

# Returns:
- Symbol matrix diagonal for the operator

# Example:
```jldoctest
# setup
mesh = Mesh1D(1.0);
diffusion = GalleryOperator("diffusion", 3, 3, mesh);

# note: either syntax works
diagonal = LFAToolkit.getdiagonal(diffusion);
diagonal = diffusion.diagonal;

# verify
@assert diagonal ≈ [7/6 0; 0 4/3]

# output

```
"""
function getdiagonal(operator::Operator)
    # assemble if needed
    if !isdefined(operator, :diagonal)
        # setup
        rowmodemap = operator.rowmodemap
        columnmodemap = operator.columnmodemap
        elementmatrix = operator.elementmatrix

        # compute
        diagonalnodes = Diagonal(elementmatrix)
        diagonalmodes = Diagonal(rowmodemap * diagonalnodes * columnmodemap)

        # store
        operator.diagonal = diagonalmodes
    end

    # return
    return getfield(operator, :diagonal)
end

"""
```julia
getmultiplicity(operator)
```

Compute or retrieve the vector of node multiplicity for the operator

# Arguments:
- `operator`:  operator to compute node multiplicity

# Returns:
- Vector of node multiplicity for the operator

# Example
```jldoctest
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
    diffusion = GalleryOperator("diffusion", 5, 5, mesh);

    # compute multiplicity
    multiplicity = diffusion.multiplicity;

    # verify
    multiplicity1D = [2. 1. 1. 1. 2.]';
    if dimension == 1
        @assert multiplicity ≈ multiplicity1D
    elseif dimension == 2
        @assert multiplicity ≈ kron(multiplicity1D, multiplicity1D)
    elseif dimension == 3
        @assert multiplicity ≈ kron(multiplicity1D, multiplicity1D, multiplicity1D)
    end
end
"""
function getmultiplicity(operator::Operator)
    # assemble if needed
    if !isdefined(operator, :multiplicity)
        # fill matrix
        numbernodes = size(operator.elementmatrix)[2]
        multiplicity_sum = spzeros(numbernodes)

        # count multiplicity
        currentnode = 0
        for input in operator.inputs
            if input.evaluationmodes[1] != EvaluationMode.quadratureweights
                for i = 1:input.basis.numbernodes*input.basis.numbercomponents
                    multiplicity_sum[input.basis.modemap[i]+currentnode] += 1
                end
                currentnode += input.basis.numbernodes
            end
        end

        # update shared nodes
        currentnode = 0
        multiplicity = spzeros(numbernodes)
        for input in operator.inputs
            if input.evaluationmodes[1] != EvaluationMode.quadratureweights
                for i = 1:input.basis.numbernodes*input.basis.numbercomponents
                    multiplicity[i+currentnode] =
                        multiplicity_sum[input.basis.modemap[i]+currentnode]
                end
                currentnode += input.basis.numbernodes
            end
        end

        # store
        operator.multiplicity = multiplicity
    end

    # return
    return getfield(operator, :multiplicity)
end

"""
```julia
getrowmodemap(operator)
```

Compute or retrieve the matrix mapping the rows of the element matrix to the symbol matrix

# Arguments:
- `operator`:  operator to compute row mode map

# Returns:
- Matrix mapping rows of element matrix to symbol matrix

# Example:
```jldoctest
# setup
mesh = Mesh1D(1.0);
mass = GalleryOperator("mass", 4, 4, mesh);

# note: either syntax works
modemap = LFAToolkit.getrowmodemap(mass);
modemap = mass.rowmodemap;

# verify
@assert modemap ≈ [1 0 0 1; 0 1 0 0; 0 0 1 0]

# output

```
"""
function getrowmodemap(operator::Operator)
    # assemble if needed
    if !isdefined(operator, :rowmodemap)
        # count modes
        numbermodes = 0
        for output in operator.outputs
            if output.evaluationmodes[1] != EvaluationMode.quadratureweights
                numbermodes += output.basis.numbermodes
            end
        end

        # fill matrix
        numbercolumns = size(operator.elementmatrix)[2]
        rowmodemap = spzeros(numbermodes, numbercolumns)
        currentnode = 0
        currentmode = 0
        for output in operator.outputs
            if output.evaluationmodes[1] != EvaluationMode.quadratureweights
                for i = 1:output.basis.numbernodes*output.basis.numbercomponents
                    rowmodemap[output.basis.modemap[i]+currentmode, i+currentnode] = 1
                end
                currentnode += output.basis.numbernodes
                currentmode += output.basis.numbermodes
            end
        end

        # store
        operator.rowmodemap = rowmodemap
    end

    # return
    return getfield(operator, :rowmodemap)
end

"""
```julia
getcolumnmodemap(operator)
```

Compute or retrieve the matrix mapping the columns of the element matrix to the
  symbol matrix

# Arguments:
- `operator`:  operator to compute column mode map

# Returns:
- Matrix mapping columns of element matrix to symbol matrix

# Example:
```jldoctest
# setup
mesh = Mesh1D(1.0);
mass = GalleryOperator("mass", 4, 4, mesh);

# note: either syntax works
modemap = LFAToolkit.getcolumnmodemap(mass);
modemap = mass.columnmodemap;

# verify
@assert modemap ≈ [1 0 0; 0 1 0; 0 0 1; 1 0 0]

# output

```
"""
function getcolumnmodemap(operator::Operator)
    # assemble if needed
    if !isdefined(operator, :columnmodemap)
        # count modes
        numbermodes = 0
        for input in operator.inputs
            if input.evaluationmodes[1] != EvaluationMode.quadratureweights
                numbermodes += input.basis.numbermodes
            end
        end

        # fill matrix
        numberrows = size(operator.elementmatrix)[2]
        columnmodemap = spzeros(numberrows, numbermodes)
        currentnode = 0
        currentmode = 0
        for input in operator.inputs
            if input.evaluationmodes[1] != EvaluationMode.quadratureweights
                for i = 1:input.basis.numbernodes*input.basis.numbercomponents
                    columnmodemap[i+currentnode, input.basis.modemap[i]+currentmode] = 1
                end
                currentnode += input.basis.numbernodes
                currentmode += input.basis.numbermodes
            end
        end

        # store
        operator.columnmodemap = columnmodemap
    end

    # return
    return getfield(operator, :columnmodemap)
end

"""
```julia
getinputcoordinates(operator)
```

Compute or retrieve the array of input coordinates

# Arguments:
- `operator`:  operator to get input coordinates

# Returns:
- Array of input coordinates
"""
function getinputcoordinates(operator::Operator)
    # assemble if needed
    if !isdefined(operator, :inputcoordinates)
        # setup for computation
        inputcoordinates = []
        for input in operator.inputs
            if input.evaluationmodes[1] != EvaluationMode.quadratureweights
                currentinputnodes = kron(
                    ones(input.basis.numbercomponents, input.basis.numbercomponents),
                    input.basis.nodes,
                )
                inputcoordinates =
                    inputcoordinates == [] ? currentinputnodes :
                    [inputcoordinates; currentinputnodes]
            end
        end

        # store
        operator.inputcoordinates = inputcoordinates
    end

    # return
    return getfield(operator, :inputcoordinates)
end

"""
```julia
getoutputcoordinates(operator)
```

Compute or retrieve the array of output coordinates

# Arguments:
- `operator`:  operator to get output coordinates

# Returns:
- Array of output coordinates
"""
function getoutputcoordinates(operator::Operator)
    # assemble if needed
    if !isdefined(operator, :outputcoordinates)
        # setup for computation
        outputcoordinates = []
        for output in operator.outputs
            currentoutputnodes = kron(
                ones(output.basis.numbercomponents, output.basis.numbercomponents),
                output.basis.nodes,
            )
            outputcoordinates =
                outputcoordinates == [] ? currentoutputnodes :
                [outputcoordinates; currentoutputnodes]
        end

        # store
        operator.outputcoordinates = outputcoordinates
    end

    # return
    return getfield(operator, :outputcoordinates)
end

"""
```julia
getnodecoordinatedifferences(operator)
```

Compute or retrieve the array of differences in coordinates between nodes

# Arguments:
- `operator`:  operator to compute differences between node coordinates

# Returns:
- Array of differences in coordinates between nodes

# Example:
```jldoctest
# setup
mesh = Mesh1D(1.0);
mass = GalleryOperator("mass", 4, 4, mesh);

# note: either syntax works
nodedifferences = LFAToolkit.getnodecoordinatedifferences(mass);
nodedifferences = mass.nodecoordinatedifferences;

# verify
truenodes = LFAToolkit.gausslobattoquadrature(4, false);
truenodedifferences = [
    (truenodes[j] - truenodes[i])/2.0 for i in 1:4, j in 1:4
];
@assert nodedifferences ≈ truenodedifferences

# output

```
"""
function getnodecoordinatedifferences(operator::Operator)
    # assemble if needed
    if !isdefined(operator, :nodecoordinatedifferences)
        # setup for computation
        inputcoordinates = operator.inputcoordinates
        outputcoordinates = operator.outputcoordinates
        dimension = operator.inputs[1].basis.dimension
        lengths = [
            maximum(inputcoordinates[:, d]) - minimum(inputcoordinates[:, d]) for
            d = 1:dimension
        ]

        # fill matrix
        numberrows, numbercolumns = size(operator.elementmatrix)
        nodecoordinatedifferences = zeros(numberrows, numbercolumns, dimension)
        @show size(inputcoordinates)
        @show inputcoordinates
        @show size(outputcoordinates)
        @show outputcoordinates
        @show size(nodecoordinatedifferences)
        for i = 1:numberrows, j = 1:numbercolumns, k = 1:dimension
            nodecoordinatedifferences[i, j, k] =
                (inputcoordinates[j, k] - outputcoordinates[i, k]) / lengths[k]
        end

        # store
        operator.nodecoordinatedifferences = nodecoordinatedifferences
    end

    # return
    return getfield(operator, :nodecoordinatedifferences)
end

# ------------------------------------------------------------------------------
# get/set property
# ------------------------------------------------------------------------------

function Base.getproperty(operator::Operator, f::Symbol)
    if f == :dimension
        return getdimension(operator)
    elseif f == :elementmatrix
        return getelementmatrix(operator)
    elseif f == :diagonal
        return getdiagonal(operator)
    elseif f == :multiplicity
        return getmultiplicity(operator)
    elseif f == :rowmodemap
        return getrowmodemap(operator)
    elseif f == :columnmodemap
        return getcolumnmodemap(operator)
    elseif f == :inputcoordinates
        return getinputcoordinates(operator)
    elseif f == :outputcoordinates
        return getoutputcoordinates(operator)
    elseif f == :nodecoordinatedifferences
        return getnodecoordinatedifferences(operator)
    else
        return getfield(operator, f)
    end
end

function Base.setproperty!(operator::Operator, f::Symbol, value)
    if f == :weakform || f == :mesh || f == :inputs || f == :outputs
        throw(ReadOnlyMemoryError()) # COV_EXCL_LINE
    else
        return setfield!(operator, f, value)
    end
end

# ------------------------------------------------------------------------------
# compute symbol matrix
# ------------------------------------------------------------------------------

"""
```julia
computesymbols(operator, θ)
```

Compute the symbol matrix for an operator

# Arguments:
- `operator`:  finite element operator to compute symbol matrix for
- `θ`:         Fourier mode frequency array (one frequency per dimension)

# Returns:
- Symbol matrix for the operator

# Example:
```jldoctest
using LinearAlgebra;

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
    A = computesymbols(diffusion, π*ones(dimension));

    # verify
    eigenvalues = real(eigvals(A));
    if dimension == 1
        @assert minimum(eigenvalues) ≈ 1
        @assert maximum(eigenvalues) ≈ 4/3
    elseif dimension == 2
        @assert minimum(eigenvalues) ≈ 2/3
        @assert maximum(eigenvalues) ≈ 64/45
    elseif dimension == 3
        @assert minimum(eigenvalues) ≈ 1/3
        @assert maximum(eigenvalues) ≈ 256/225
    end
end

# output

```
"""
function computesymbols(operator::Operator, θ::Array)
    # validity check
    dimension = length(θ)
    if dimension != operator.inputs[1].basis.dimension
        throw(ArgumentError("Must provide as many values of θ as the mesh has dimensions")) # COV_EXCL_LINE
    end

    # setup
    rowmodemap = operator.rowmodemap
    columnmodemap = operator.columnmodemap
    elementmatrix = operator.elementmatrix
    numberrows, numbercolumns = size(elementmatrix)
    nodecoordinatedifferences = operator.nodecoordinatedifferences
    symbolmatrixnodes = zeros(ComplexF64, numberrows, numbercolumns)

    # compute
    for i = 1:numberrows, j = 1:numbercolumns
        symbolmatrixnodes[i, j] =
            elementmatrix[i, j] *
            ℯ^(im * sum([θ[k] * nodecoordinatedifferences[i, j, k] for k = 1:dimension]))
    end
    symbolmatrixmodes = rowmodemap * symbolmatrixnodes * columnmodemap

    # return
    return symbolmatrixmodes
end

# ------------------------------------------------------------------------------
