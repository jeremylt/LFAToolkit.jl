# ------------------------------------------------------------------------------
# bddc
# ------------------------------------------------------------------------------

"""
```julia
BDDC(fineoperator, coarseoperator, smoother, prolongation)
```

BDDC preconditioner for finite element operators

# Arguments:
- `fineoperator`:  finite element operator to precondition
- `injectiontype`: type of injection into subassembled space to use

# Returns:
- BDDC preconditioner object
"""
mutable struct BDDC <: AbstractPreconditioner
    # data never changed
    fineoperator::Operator
    injectiontype::BDDCInjectionType.BDDCInjectType

    # data empty until assembled
    prolongationmatrix::AbstractArray{Float64}
    nodecoordinatedifferences::AbstractArray{Float64}

    # inner constructor
    BDDC(fineoperator::Operator, injectiontype::BDDCInjectionType.BDDCInjectType) = (
    # constructor
    new(fineoperator, injectiontype))
end

# printing
# COV_EXCL_START
function Base.show(io::IO, preconditioner::BDDC)
    if preconditioner.injectiontype == BDDCInjectionType.scaled
        print(io, "lumped ")
    elseif preconditioner.injectiontype == BDDCInjectionType.harmonic
        print(io, "Dirichlet ")
    end
    print(io, "BDDC preconditioner")
end
# COV_EXCL_STOP

# ------------------------------------------------------------------------------
