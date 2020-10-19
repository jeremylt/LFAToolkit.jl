# ------------------------------------------------------------------------------
# P-multigrid
# ------------------------------------------------------------------------------

"""
```julia
PMultigrid(fineoperator, coarseoperator, smoother, prolongation)
```

P-Multigrid preconditioner for finite element operators

# Arguments:
- `fineoperator`:      finite element operator to precondition
- `coarseoperator`:    coarse grid representation of finite element operator to precondition
- `smoother`:          error relaxation operator, such as Jacobi
- `prolongationbasis`: element prolongation matrix from coarse to fine grid

# Returns:
- P-multigrid preconditioner object

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
finebasis = TensorH1LagrangeBasis(4, 4, 2);
coarsebasis = TensorH1LagrangeBasis(2, 4, 2);
collocatedquadrature = true;
ctofbasis = TensorH1LagrangeBasis(2, 4, 2, collocatedquadrature);
 
# diffusion setup
function diffusionweakform(du::Array{Float64}, w::Array{Float64})
    dv = du*w[1]
    return [dv]
end
    
# diffusion operator, fine grid
fineinputs = [
    OperatorField(finebasis, [EvaluationMode.gradient]),
    OperatorField(finebasis, [EvaluationMode.quadratureweights]),
];
fineoutputs = [OperatorField(finebasis, [EvaluationMode.gradient])];
finediffusion = Operator(diffusionweakform, mesh, fineinputs, fineoutputs);

# diffusion operator, coarse grid
coarseinputs = [
    OperatorField(coarsebasis, [EvaluationMode.gradient]),
    OperatorField(coarsebasis, [EvaluationMode.quadratureweights]),
];
coarseoutputs = [OperatorField(coarsebasis, [EvaluationMode.gradient])];
coarsediffusion = Operator(diffusionweakform, mesh, coarseinputs, coarseoutputs);

# smoother
jacobi = Jacobi(finediffusion);

# preconditioner
multigrid = PMultigrid(finediffusion, coarsediffusion, jacobi, ctofbasis);

# verify
println(multigrid)
println(multigrid.fineoperator)

# output
p-multigrid preconditioner
finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    dimension: 2
  evaluation mode:
    gradient
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
    gradient
```
"""
mutable struct PMultigrid <: AbstractPreconditioner
    # data never changed
    fineoperator::Operator
    coarseoperator::Any
    smoother::AbstractPreconditioner
    prolongationbasis::AbstractBasis

    # data empty until assembled
    nodecoordinatedifferences::AbstractArray{Float64}

    # inner constructor
    PMultigrid(fineoperator, coarseoperator, smoother, prolongationbasis) = (
        # check smoother for fine grid
        if fineoperator != smoother.operator
            error("smoother must be for fine grid operator") # COV_EXCL_LINE
        end;

        # check coarse operator is operator or Multigrid
        if !isa(coarseoperator, Operator) && !isa(coarseoperator, PMultigrid)
            error("coarse operator must be an operator or multigrid") # COV_EXCL_LINE
        end;

        # check dimensions
        if fineoperator.inputs[1].basis.dimension != prolongationbasis.dimension
            error("fine grid and prolongation space dimensions must agree") #COV_EXCL_LINE
        end;

        # constructor
        new(fineoperator, coarseoperator, smoother, prolongationbasis)
    )
end

# printing
# COV_EXCL_START
Base.show(io::IO, preconditioner::PMultigrid) = print(io, "p-multigrid preconditioner")
# COV_EXCL_STOP

# ------------------------------------------------------------------------------
# data for computing symbols
# ------------------------------------------------------------------------------

function getnodecoordinatedifferences(multigrid::PMultigrid)
    # assemble if needed
    if !isdefined(multigrid, :nodecoordinatedifferences)
        # setup for computation
        dimension = multigrid.prolongationbasis.dimension
        inputcoordinates = multigrid.coarseoperator.inputcoordinates
        outputcoordinates = multigrid.fineoperator.outputcoordinates
        lengths = [
            max(inputcoordinates[:, d]...) - min(inputcoordinates[:, d]...)
            for d = 1:dimension
        ]

        # fill matrix
        numberrows = size(outputcoordinates)[1]
        numbercolumns = size(inputcoordinates)[2]
        nodecoordinatedifferences = zeros(numberrows, numbercolumns, dimension)
        for i = 1:numberrows, j = 1:numbercolumns, k = 1:dimension
            nodecoordinatedifferences[i, j, k] =
                (inputcoordinates[j, k] - outputcoordinates[i, k])/lengths[k]
        end

        # store
        multigrid.nodecoordinatedifferences = nodecoordinatedifferences
    end

    # return
    return getfield(multigrid, :nodecoordinatedifferences)
end

function computesymbolspplongation(multigrid::PMultigrid, θ::Array)
    # setup
    rowmodemap = multigrid.fineoperator.rowmodemap
    columnmodemap = multigrid.coarseoperator.columnmodemap
    prolongationmatrix = multigrid.prolongationbasis.interpolation
    numberrows, numbercolumns = size(prolongationmatrix)
    nodecoordinatedifferences = multigrid.nodecoordinatedifferences
    symbolmatrixnodes = zeros(ComplexF64, numberrows, numbercolumns)

    # compute
    if dimension == 1
        for i = 1:numberrows, j = 1:numbercolumns
            symbolmatrixnodes[i, j] =
                prolongationmatrix[i, j]*ℯ^(im*θ[1]*nodecoordinatedifferences[i, j, 1])
        end
    elseif dimension == 2
        for i = 1:numberrows, j = 1:numbercolumns
            symbolmatrixnodes[i, j] =
                prolongationmatrix[
                    i,
                    j,
                ]*ℯ^(
                    im*(
                        θ[1]*nodecoordinatedifferences[i, j, 1] +
                        θ[2]*nodecoordinatedifferences[i, j, 2]
                    )
                )
        end
    elseif dimension == 3
        for i = 1:numberrows, j = 1:numbercolumns
            symbolmatrixnodes[i, j] =
                prolongationmatrix[
                    i,
                    j,
                ]*ℯ^(
                    im*(
                        θ[1]*nodecoordinatedifferences[i, j, 1] +
                        θ[2]*nodecoordinatedifferences[i, j, 2] +
                        θ[3]*nodecoordinatedifferences[i, j, 3]
                    )
                )
        end
    end
    symbolmatrixmodes = rowmodemap*symbolmatrixnodes*columnmodemap

    # return
    return symbolmatrixmodes
end

# ------------------------------------------------------------------------------
# get/set property
# ------------------------------------------------------------------------------

function Base.getproperty(multigrid::PMultigrid, f::Symbol)
    if f == :nodecoordinatedifferences
        return getnodecoordinatedifferences(multigrid)
    else
        return getfield(multigrid, f)
    end
end

function Base.setproperty!(multigrid::PMultigrid, f::Symbol, value)
    if f == :fineoperator ||
       f == :coarseoperator ||
       f == :smoother ||
       f == :prolongationbasis
        throw(ReadOnlyMemoryError()) # COV_EXCL_LINE
    else
        return setfield!(multigrid, f, value)
    end
end

# ------------------------------------------------------------------------------
# compute symbols
# ------------------------------------------------------------------------------

function computesymbols(multigrid::PMultigrid, p::Array, θ::Array)
    # compute components
    S_f = computesymbols(multigrid.smoother, p, θ)

    P_ctof = computesymbolspprolongation(multigrid, θ)
    R_ftoc = P^T

    A_f = computesymbols(multigrid.fineoperator, θ)
    A_c = []
    if isa(multigrid.coarseoperator, Operator)
        computesymbols(multigrid.coarseoperator, θ)
    elseif isa(multigrid.coarseoperator, PMultigrid)
        computesymbols(multigrid.coarseoperator, p, θ)
    else
        Throw(error("coarse operator not supported")) # COV_EXCL_LINE
    end

    # return
    return S_f*(I - P_ctof*A_C^-1*R_ftoc*A_f)*S_f
end

# ------------------------------------------------------------------------------
