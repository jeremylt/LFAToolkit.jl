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
- `coarseoperator`:    coarse grid representation of finite element operator to
                           precondition
- `smoother`:          error relaxation operator, such as Jacobi
- `prolongationbases`: element prolongation bases from coarse to fine grid

# Returns:
- P-multigrid preconditioner object

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
finebasis = TensorH1LagrangeBasis(5, 5, 2);
coarsebasis = TensorH1LagrangeBasis(3, 5, 2);
ctofbasis = TensorH1LagrangeBasis(3, 5, 2, lagrangequadrature=true);
 
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
multigrid = PMultigrid(finediffusion, coarsediffusion, jacobi, [ctofbasis]);

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
    numbernodes1d: 5
    numberquadraturepoints1d: 5
    dimension: 2
  evaluation mode:
    gradient
operator field:
tensor product basis:
    numbernodes1d: 5
    numberquadraturepoints1d: 5
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
tensor product basis:
    numbernodes1d: 5
    numberquadraturepoints1d: 5
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
    prolongationbases::Array{AbstractBasis}

    # data empty until assembled
    prolongationmatrix::AbstractArray{Float64}
    nodecoordinatedifferences::AbstractArray{Float64}

    # inner constructor
    PMultigrid(fineoperator, coarseoperator, smoother, prolongationbases) = (
        # check smoother for fine grid
        if fineoperator != smoother.operator
            error("smoother must be for fine grid operator") # COV_EXCL_LINE
        end;

        # check coarse operator is operator or Multigrid
        if !isa(coarseoperator, Operator) && !isa(coarseoperator, PMultigrid)
            error("coarse operator must be an operator or multigrid") # COV_EXCL_LINE
        end;

        # check agreement in number of fields
        if length(prolongationbases) != length(fineoperator.outputs) ||
           length(prolongationbases) != length(coarseoperator.outputs)
            error("operators and prolongation bases must have same number of fields") # COV_EXCL_LINE
        end;

        # check dimensions
        for basis in prolongationbases
            if fineoperator.inputs[1].basis.dimension != basis.dimension
                error("fine grid and prolongation space dimensions must agree") #COV_EXCL_LINE
            end
        end;

        # constructor
        new(fineoperator, coarseoperator, smoother, prolongationbases)
    )
end

# printing
# COV_EXCL_START
Base.show(io::IO, preconditioner::PMultigrid) = print(io, "p-multigrid preconditioner")
# COV_EXCL_STOP

# ------------------------------------------------------------------------------
# data for computing symbols
# ------------------------------------------------------------------------------

"""
```julia
getnodecoordinateddifferences(multigrid)
```

Compute or retrieve the array of differences in coordinates between nodes

# Returns:
- Array of differences in coordinates between nodes
"""
function getnodecoordinatedifferences(multigrid::PMultigrid)
    # assemble if needed
    if !isdefined(multigrid, :nodecoordinatedifferences)
        # setup for computation
        dimension = multigrid.prolongationbases[1].dimension
        inputcoordinates = multigrid.coarseoperator.inputcoordinates
        outputcoordinates = multigrid.fineoperator.outputcoordinates
        lengths = [
            max(inputcoordinates[:, d]...) - min(inputcoordinates[:, d]...)
            for d = 1:dimension
        ]

        # fill matrix
        numberrows = size(outputcoordinates)[1]
        numbercolumns = size(inputcoordinates)[1]
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


"""
```julia
getprolongationmatrix(multigrid)
```

Compute or retrieve the prolongation matrix

# Returns:
- Matrix prolonging from coarse grid to fine grid
"""
function getprolongationmatrix(multigrid::PMultigrid)
    # assemble if needed
    if !isdefined(multigrid, :prolongationmatrix)
        # setup
        numberfinenodes = 0
        numbercoarsenodes = 0
        Pblocks = []

        # field prolongation matrices
        for basis in multigrid.prolongationbases
            # number of nodes
            numberfinenodes += basis.numberquadraturepoints
            numbercoarsenodes += basis.numbernodes

            # prolongation matrices
            push!(Pblocks, basis.interpolation)
        end

        prolongationmatrix = spzeros(numberfinenodes, numbercoarsenodes)
        currentrow = 1
        currentcolumn = 1
        for Pblock in Pblocks
            prolongationmatrix[currentrow:size(Pblock)[1], currentcolumn:size(Pblock)[2]] =
                Pblock
            currentrow += size(Pblock)[1]
            currentcolumn += size(Pblock)[2]
        end

        # store
        multigrid.prolongationmatrix = prolongationmatrix
    end

    # return
    return getfield(multigrid, :prolongationmatrix)
end

"""
```julia
computesymbolspprolongation(multigrid, θ)
```

Compute the symbol matrix for a p-multigrid prolongation operator

# Arguments:
- `multigrid`: P-multigrid operator to compute prolongation symbol matrix for
- `θ`:         Fourier mode frequency array (one frequency per dimension)

# Returns:
- Symbol matrix for the p-multigrid prolongation operator
"""
function computesymbolspprolongation(multigrid::PMultigrid, θ::Array)
    # setup
    dimension = multigrid.prolongationbases[1].dimension
    rowmodemap = multigrid.fineoperator.rowmodemap
    columnmodemap = multigrid.coarseoperator.columnmodemap
    prolongationmatrix = multigrid.prolongationmatrix
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
    if f == :prolongationmatrix
        return getprolongationmatrix(multigrid)
    elseif f == :nodecoordinatedifferences
        return getnodecoordinatedifferences(multigrid)
    elseif f == :columnmodemap # Used if nesting multigrid levels
        return multigrid.fineoperator.columnmodemap
    elseif f == :inputcoordinates # Used if nesting multigrid levels 
        return multigrid.fineoperator.inputcoordinates
    else
        return getfield(multigrid, f)
    end
end

function Base.setproperty!(multigrid::PMultigrid, f::Symbol, value)
    if f == :fineoperator ||
       f == :coarseoperator ||
       f == :smoother ||
       f == :prolongationbases
        throw(ReadOnlyMemoryError()) # COV_EXCL_LINE
    else
        return setfield!(multigrid, f, value)
    end
end

# ------------------------------------------------------------------------------
# compute symbols
# ------------------------------------------------------------------------------

"""
```julia
computesymbols(multigrid, p, v, θ)
```

Compute or retrieve the symbol matrix for a Jacobi preconditioned operator

# Arguments:
- `multigrid`: PMultigrid preconditioner to compute symbol matrix for
- `p`:         Smoothing paramater array
- `v`:         Number of pre and post smooths
- `θ`:         Fourier mode frequency array (one frequency per dimension)

# Returns:
- Symbol matrix for the p-multigrid preconditioned operator

# Example:
```jldoctest
using LinearAlgebra

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
    finebasis = TensorH1LagrangeBasis(5, 5, dimension);
    coarsebasis = TensorH1LagrangeBasis(3, 5, dimension);
    ctofbasis = TensorH1LagrangeBasis(3, 5, dimension, lagrangequadrature=true);
    
    
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
    multigrid = PMultigrid(finediffusion, coarsediffusion, jacobi, [ctofbasis]);

    # compute symbols
    A = computesymbols(multigrid, [1.0], [1, 1], π*ones(dimension));

    # verify
    using LinearAlgebra;
    eigenvalues = real(eigvals(A));
    if dimension == 1
       @assert min(eigenvalues...) ≈ -3.8926063079259547
       @assert max(eigenvalues...) ≈ 0.6907788855328606
    elseif dimension == 2
       @assert min(eigenvalues...) ≈ -1.5478589373279685
       @assert max(eigenvalues...) ≈ 0.995724230621591
    elseif dimension == 3
       @assert min(eigenvalues...) ≈ -1.8934899363339088
       @assert max(eigenvalues...) ≈ 1.4306256666321246
    end
end

# output

```
"""
function computesymbols(multigrid::PMultigrid, p::Array, v::Array{Int}, θ::Array)
    # validate number of parameters
    if length(v) != 2
        Throw(error("must specify number of pre and post smooths")) # COV_EXCL_LINE
    end

    # compute component symbols
    S_f = computesymbols(multigrid.smoother, p, θ)

    P_ctof = computesymbolspprolongation(multigrid, θ)
    R_ftoc = transpose(P_ctof)

    A_f = computesymbols(multigrid.fineoperator, θ)
    A_c = []
    if isa(multigrid.coarseoperator, Operator)
        A_c = computesymbols(multigrid.coarseoperator, θ)
    elseif isa(multigrid.coarseoperator, PMultigrid)
        A_c = computesymbols(multigrid.coarseoperator, p, v, θ)
    else
        Throw(error("coarse operator not supported")) # COV_EXCL_LINE
    end

    # return
    return S_f^v[2]*(I - P_ctof*A_c^-1*R_ftoc*A_f)*S_f^v[1]
end

# ------------------------------------------------------------------------------
