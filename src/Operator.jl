# ---------------------------------------------------------------------------------------------------------------------
# finite element operators
# ---------------------------------------------------------------------------------------------------------------------

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
- `weakform`: user provided function that represents weak form at quadrature points
- `mesh`:     mesh object with deformation in each dimension
- `inputs`:   array of operator input fields
- `outputs`:  array of operator output fields

# Returns:
- Finite element operator object

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
basis = TensorH1LagrangeBasis(4, 4, 2);
    
function massweakform(u::Array{Float64}, w::Array{Float64})
    v = u * w[1]
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
@assert mass.weakform == massweakform
@assert mass.mesh == mesh

# output
finite element operator:
2D mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    dimension: 2
  evaluation mode:
    interpolation
operator field:
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    dimension: 2
  evaluation mode:
    interpolation
```
"""
mutable struct Operator
    # data never changed
    weakform::Function
    mesh::Mesh
    inputs::Array{OperatorField}
    outputs::Array{OperatorField}

    # data empty until assembled
    elementmatrix::Array{Float64,2}
    rowmodemap::Array{Float64,2}
    columnmodemap::Array{Float64,2}
    nodecoordinatedifferences::Array{Float64}

    # inner constructor
    Operator(weakform, mesh, inputs, outputs) = (dimension = 0;
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

    # constructor
    new(weakform, mesh, inputs, outputs))
end

function Base.show(io::IO, operator::Operator)
    print(io, "finite element operator:\n", operator.mesh)

    # inputs
    if length(operator.inputs) == 1
        print("\n\n1 input:")
    else
        print("\n\n", length(operator.inputs), " inputs:")
    end
    for i = 1:length(operator.inputs)
        print("\n", operator.inputs[i])
    end

    # outputs
    if length(operator.outputs) == 1
        print("\n\n1 output:")
    else
        print("\n\n", length(operator.outputs), " outputs:")
    end
    for i = 1:length(operator.outputs)
        print("\n", operator.outputs[i])
    end
end

# ---------------------------------------------------------------------------------------------------------------------
# data for computing symbols
# ---------------------------------------------------------------------------------------------------------------------

"""
```julia
getelementmatrix(operator)
```

Compute or retrieve the element matrix of operator for computing the symbol

# Arguments:
- `operator`: operator to compute element element matrix

# Returns:
- Assembled element matrix

# Mass matrix example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
basis = TensorH1LagrangeBasis(4, 4, 2);
    
function massweakform(u::Array{Float64}, w::Array{Float64})
    v = u * w[1]
    return [v]
end
    
# mass operator
inputs = [
    OperatorField(basis, [EvaluationMode.interpolation]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.interpolation])];
mass = Operator(massweakform, mesh, inputs, outputs);
    
# element matrix computation
# note: either syntax works
elementmatrix = LFAToolkit.getelementmatrix(mass);
elementmatrix = mass.elementmatrix;

# verify
u = ones(4*4);
v = elementmatrix * u;
    
total = sum(v);
@assert total ≈ 4.0
    
# output

```

# Diffusion matrix example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
basis = TensorH1LagrangeBasis(4, 4, 2);
    
function diffusionweakform(du::Array{Float64}, w::Array{Float64})
    dv = du * w[1]
    return [dv]
end
    
# diffusion operator
inputs = [
    OperatorField(basis, [EvaluationMode.gradient]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.gradient])];
diffusion = Operator(diffusionweakform, mesh, inputs, outputs);
    
# element matrix computation
# note: either syntax works
elementmatrix = LFAToolkit.getelementmatrix(diffusion);
elementmatrix = diffusion.elementmatrix;
    
# verify
u = ones(4*4);
v = elementmatrix * u;
    
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
        numberquadraturepoints = 0
        numberquadratureinputs = 0
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
                numbermodes = 0
                Bcurrent = []
                for mode in input.evaluationmodes
                    if mode == EvaluationMode.interpolation
                        numbermodes += 1
                        numberquadratureinputs += 1
                        Bcurrent =
                            Bcurrent == [] ? input.basis.interpolation :
                            [Bcurrent; input.basis.interpolation]
                    elseif mode == EvaluationMode.gradient
                        numbermodes += input.basis.dimension
                        numberquadratureinputs += input.basis.dimension
                        Bcurrent =
                            Bcurrent == [] ? input.basis.gradient :
                            [Bcurrent; input.basis.gradient]
                    end
                end
                push!(Bblocks, Bcurrent)
                push!(weakforminputs, zeros(numbermodes))
            end
        end

        # input basis matrix
        B = spzeros(numberquadratureinputs * numberquadraturepoints, numbernodes)
        currentrow = 1
        currentcolumn = 1
        for Bblock in Bblocks
            B[currentrow:size(Bblock)[1], currentcolumn:size(Bblock)[2]] = Bblock
            currentrow += size(Bblock)[1]
            currentcolumn += size(Bblock)[2]
        end

        # quadrature weight input index
        if weightinputindex != 0
            quadratureweights =
                getquadratureweights(operator.inputs[weightinputindex].basis)
        end

        # outputs
        for output in operator.outputs
            # output evaluation modes
            numbermodes = 0
            Btcurrent = []
            for mode in output.evaluationmodes
                if mode == EvaluationMode.interpolation
                    Btcurrent =
                        Btcurrent == [] ? output.basis.interpolation :
                        [Btcurrent; output.basis.intepolation]
                elseif mode == EvaluationMode.gradient
                    Btcurrent =
                        Btcurrent == [] ? output.basis.gradient :
                        [Btcurrent; output.basis.gradient]
                    # note: quadrature weights checked in constructor
                end
            end
            push!(Btblocks, Btcurrent)
        end

        # output basis matrix
        Bt = spzeros(numberquadratureinputs * numberquadraturepoints, numbernodes)
        currentrow = 1
        currentcolumn = 1
        for Btblock in Btblocks
            Bt[currentrow:size(Btblock)[1], currentcolumn:size(Btblock)[2]] = Btblock
            currentrow += size(Btblock)[1]
            currentcolumn += size(Btblock)[2]
        end
        Bt = transpose(Bt)

        # QFunction matrix
        D = spzeros(
            numberquadratureinputs * numberquadraturepoints,
            numberquadratureinputs * numberquadraturepoints,
        )
        for q = 1:numberquadraturepoints
            # set quadrature weight
            if weightinputindex != 0
                weakforminputs[weightinputindex][1] = quadratureweights[q]
            end

            # loop over inputs
            currentfieldin = 0
            for i = 1:length(operator.inputs)
                input = operator.inputs[i]
                if input.evaluationmodes[1] == EvaluationMode.quadratureweights
                    break
                end

                # fill sparse matrix
                for j = 1:length(input.evaluationmodes)
                    # run user weak form function
                    weakforminputs[i][j] = 1.0
                    outputs = operator.weakform(weakforminputs...)
                    weakforminputs[i][j] = 0.0

                    # store outputs
                    currentfieldout = 0
                    for k = 1:length(operator.outputs)
                        output = operator.outputs[k]
                        for l = 1:length(output.evaluationmodes)
                            D[
                                currentfieldout*numberquadraturepoints+q,
                                currentfieldin*numberquadraturepoints+q,
                            ] = outputs[k][l]
                            currentfieldout += 1
                        end
                    end
                end
            end
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
getrowmodemap()
```

Compute or retrieve the matrix mapping the rows of the element matrix to the symbol matrix

# Returns:
- Matrix mapping rows of element matrix to symbol matrix

# Example:
```jldoctest
# setup
mesh = Mesh1D(1.0);
basis = TensorH1LagrangeBasis(4, 4, 1);
    
function massweakform(u::Array{Float64}, w::Array{Float64})
    v = u * w[1]
    return [v]
end
    
# mass operator
inputs = [
    OperatorField(basis, [EvaluationMode.interpolation]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.interpolation])];
mass = Operator(massweakform, mesh, inputs, outputs);

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
                for i = 1:output.basis.numbernodes
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
getcolumnmodemap()
```

Compute or retrieve the matrix mapping the columns of the element matrix to the symbol matrix

# Returns:
- Matrix mapping columns of element matrix to symbol matrix

# Example:
```jldoctest
# setup
mesh = Mesh1D(1.0);
basis = TensorH1LagrangeBasis(4, 4, 1);
    
function massweakform(u::Array{Float64}, w::Array{Float64})
    v = u * w[1]
    return [v]
end
    
# mass operator
inputs = [
    OperatorField(basis, [EvaluationMode.interpolation]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.interpolation])];
mass = Operator(massweakform, mesh, inputs, outputs);

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
                for i = 1:input.basis.numbernodes
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
getnodecoordinatedifferences()
```

Compute or retrieve the array of differences in coordinates between nodes

# Returns:
- Array of differences in coordinates between nodes

# Example:
```jldoctest
# setup
mesh = Mesh1D(1.0);
basis = TensorH1LagrangeBasis(4, 4, 1);
    
function massweakform(u::Array{Float64}, w::Array{Float64})
    v = u * w[1]
    return [v]
end
    
# mass operator
inputs = [
    OperatorField(basis, [EvaluationMode.interpolation]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.interpolation])];
mass = Operator(massweakform, mesh, inputs, outputs);

# note: either syntax works
nodedifferences = LFAToolkit.getnodecoordinatedifferences(mass);
nodedifferences = mass.nodecoordinatedifferences;

# verify
truenodes = LFAToolkit.lobattoquadrature(4, false);
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
        inputcoordinates = []
        outputcoordinates = []
        for input in operator.inputs
            if input.evaluationmodes[1] != EvaluationMode.quadratureweights
                inputcoordinates =
                    inputcoordinates == [] ? input.basis.nodes :
                    [inputcoordinates; input.basis.nodes]
            end
        end
        for output in operator.outputs
            outputcoordinates =
                outputcoordinates == [] ? output.basis.nodes :
                [outputcoordinates; output.basis.nodes]
        end
        dimension = operator.inputs[1].basis.dimension
        lengths = [
            max(inputcoordinates[:, d]...) - min(inputcoordinates[:, d]...)
            for d = 1:dimension
        ]

        # fill matrix
        numberrows, numbercolumns = size(operator.elementmatrix)
        nodecoordinatedifferences = zeros(numberrows, numbercolumns, dimension)
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

# ---------------------------------------------------------------------------------------------------------------------
# get/set property
# ---------------------------------------------------------------------------------------------------------------------

function Base.getproperty(operator::Operator, f::Symbol)
    if f == :elementmatrix
        return getelementmatrix(operator)
    elseif f == :rowmodemap
        return getrowmodemap(operator)
    elseif f == :columnmodemap
        return getcolumnmodemap(operator)
    elseif f == :nodecoordinatedifferences
        return getnodecoordinatedifferences(operator)
    else
        return getfield(operator, f)
    end
end

# ---------------------------------------------------------------------------------------------------------------------
# compute symbol matrix
# ---------------------------------------------------------------------------------------------------------------------

"""
```julia
computesymbols()
```

Compute or retrieve the symbol matrix for an operator

# Returns:
- Symbol matrix for the operator

# Example:
```jldoctest
# setup
mesh = Mesh1D(1.0);
basis = TensorH1LagrangeBasis(3, 4, 1);
    
function diffusionweakform(du::Array{Float64}, w::Array{Float64})
    dv = du * w[1]
    return [dv]
end
    
# mass operator
inputs = [
    OperatorField(basis, [EvaluationMode.gradient]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.gradient])];
diffusion = Operator(diffusionweakform, mesh, inputs, outputs);

# note: either syntax works
A = computesymbols(diffusion, π);

# verify
using LinearAlgebra;
eigenvalues = eigvals(A);
@assert eigenvalues ≈ [2; 8/3]
 
# output

```
"""
function computesymbols(operator::Operator, θ_x::Number)
    # validity check
    dimension = operator.inputs[1].basis.dimension
    if dimension != 1
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
            elementmatrix[i, j] * ℯ^(im * θ_x * nodecoordinatedifferences[i, j, 1])
    end
    symbolmatrixmodes = rowmodemap * symbolmatrixnodes * columnmodemap

    # return
    return symbolmatrixmodes
end

"""
```julia
computesymbols()
```

Compute or retrieve the symbol matrix for an operator

# Returns:
- Symbol matrix for the operator

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
basis = TensorH1LagrangeBasis(3, 4, 2);
    
function diffusionweakform(du::Array{Float64}, w::Array{Float64})
    dv = du * w[1]
    return [dv]
end
    
# mass operator
inputs = [
    OperatorField(basis, [EvaluationMode.gradient]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.gradient])];
diffusion = Operator(diffusionweakform, mesh, inputs, outputs);

# note: either syntax works
A = computesymbols(diffusion, π, π);

# verify
using LinearAlgebra;
eigenvalues = eigvals(A);
@assert eigenvalues ≈ [4/3; 16/9; 64/30; 256/90]
 
# output

```
"""
function computesymbols(operator::Operator, θ_x::Number, θ_y::Number)
    # validity check
    dimension = operator.inputs[1].basis.dimension
    if dimension != 2
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
            ℯ^(
                im * (
                    θ_x * nodecoordinatedifferences[i, j, 1] +
                    θ_y * nodecoordinatedifferences[i, j, 2]
                )
            )
    end
    symbolmatrixmodes = rowmodemap * symbolmatrixnodes * columnmodemap

    # return
    return symbolmatrixmodes
end

function computesymbols(operator::Operator, θ_x::Number, θ_y::Number, θ_z::Number)
    # validity check
    dimension = operator.inputs[1].basis.dimension
    if dimension != 3
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
            ℯ^(
                im * (
                    θ_x * nodecoordinatedifferences[i, j, 1] +
                    θ_y * nodecoordinatedifferences[i, j, 2] +
                    θ_z * nodecoordinatedifferences[i, j, 3]
                )
            )
    end
    symbolmatrixmodes = rowmodemap * symbolmatrixnodes * columnmodemap

    # return
    return symbolmatrixmodes
end

# ---------------------------------------------------------------------------------------------------------------------
