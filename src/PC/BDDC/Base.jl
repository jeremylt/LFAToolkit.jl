# ------------------------------------------------------------------------------
# bddc
# ------------------------------------------------------------------------------

"""
```julia
BDDC(fineoperator, coarseoperator, smoother, prolongation)
```

BDDC preconditioner for finite element operators

# Arguments:
- `fineoperator`:      finite element operator to precondition
- `coarseoperator`:    coarse grid representation of finite element operator to
                           precondition
- `smoother`:          error relaxation operator, such as Jacobi
- `prolongationbases`: element prolongation bases from coarse to fine grid

# Returns:
- BDDC preconditioner object
"""
mutable struct BDDC <: AbstractPreconditioner
    # data never changed
    fineoperator::Operator
    coarseoperator::Any
    smoother::AbstractPreconditioner
    prolongationbases::AbstractArray{AbstractBasis}

    # data empty until assembled
    prolongationmatrix::AbstractArray{Float64}
    nodecoordinatedifferences::AbstractArray{Float64}

    # inner constructor
    BDDC(
        fineoperator::Operator,
        coarseoperator::Any,
        smoother::AbstractPreconditioner,
        prolongationbases::AbstractArray,
    ) = (
        # check smoother for fine grid
        if fineoperator != smoother.operator
            error("smoother must be for fine grid operator") # COV_EXCL_LINE
        end;

        # check coarse operator is operator or Multigrid
        if !isa(coarseoperator, Operator) && !isa(coarseoperator, Multigrid)
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
function Base.show(io::IO, preconditioner::BDDC)
    print(io, "BDDC preconditioner")
end
# COV_EXCL_STOP

# ------------------------------------------------------------------------------
