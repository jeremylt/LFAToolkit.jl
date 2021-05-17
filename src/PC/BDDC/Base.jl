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
    primalvertices::AbstractArray{Int,1}
    subassembledvertices::AbstractArray{Int,1}
    interfacevertices::AbstractArray{Int,1}
    interiorvertices::AbstractArray{Int,1}
    subassembledinverse::AbstractArray{Float64,2}
    interiorinverse::AbstractArray{Float64,2}

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
# data for computing symbols
# ------------------------------------------------------------------------------

"""
```julia
getprimalvertices(bddc)
```

Compute or retrieve the primal vertices for a BDDC preconditioner

# Returns:
- Vector of primal vertices for BDDC preconditioner

# Example:
```jldoctest
# setup
mesh = Mesh1D(1.0);
p = 4
diffusion = GalleryOperator("diffusion", p, p, mesh);
bddc = LumpedBDDC(diffusion)

# note: either syntax works
primalvertices = LFAToolkit.getprimalvertices(bddc);
primalvertices = bddc.primalvertices;

# verify
@assert primalvertices == [1, p]

# output

```
"""
function getprimalvertices(bddc::BDDC)
    # assemble if needed
    operator = bddc.fineoperator
    if !isdefined(bddc, :primalvertices)
        primalvertices = []
        numbernodes = 0
        for input in operator.inputs
            if input.evaluationmodes[1] != EvaluationMode.quadratureweights
                primalvertices =
                    vcat(primalvertices..., [input.basis.primalvertices .+ numbernodes]...)
                numbernodes += input.basis.numbernodes
            end
        end

        # store
        bddc.primalvertices = primalvertices
    end

    # return
    return getfield(bddc, :primalvertices)
end

"""
```julia
getsubassembledvertices(bddc)
```

Compute or retrieve the subassembled vertices for a BDDC preconditioner

# Returns:
- Vector of subassembled vertices for BDDC preconditioner

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
p = 4
diffusion = GalleryOperator("diffusion", p, p, mesh);
bddc = LumpedBDDC(diffusion)

# note: either syntax works
subassembledvertices = LFAToolkit.getsubassembledvertices(bddc);
subassembledvertices = bddc.subassembledvertices;

# verify
@assert subassembledvertices == setdiff(1:p^2, [1, p, p^2-p+1, p^2])

# output

```
"""
function getsubassembledvertices(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :subassembledvertices)
        numbernodes, _ = size(bddc.fineoperator.elementmatrix)
        bddc.subassembledvertices = setdiff(1:numbernodes, bddc.primalvertices)
    end

    # return
    return getfield(bddc, :subassembledvertices)
end

"""
```julia
getinterfacevertices(bddc)
```

Compute or retrieve the interface vertices for a BDDC preconditioner

# Returns:
- Vector of interface vertices for BDDC preconditioner

# Example:
```jldoctest
# setup
mesh = Mesh1D(1.0);
p = 4
diffusion = GalleryOperator("diffusion", p, p, mesh);
bddc = LumpedBDDC(diffusion)

# note: either syntax works
interfacevertices = LFAToolkit.getinterfacevertices(bddc);
interfacevertices = bddc.interfacevertices;

# verify
@assert interfacevertices == [1, p]

# output

```
"""
function getinterfacevertices(bddc::BDDC)
    # assemble if needed
    operator = bddc.fineoperator
    if !isdefined(bddc, :interfacevertices)
        interfacevertices = []
        numbernodes = 0
        for input in operator.inputs
            if input.evaluationmodes[1] != EvaluationMode.quadratureweights
                interfacevertices = vcat(
                    interfacevertices...,
                    [input.basis.interfacevertices .+ numbernodes]...,
                )
                numbernodes += input.basis.numbernodes
            end
        end

        # store
        bddc.interfacevertices = interfacevertices
    end

    # return
    return getfield(bddc, :interfacevertices)
end

"""
```julia
getinteriorvertices(bddc)
```

Compute or retrieve the interior vertices for a BDDC preconditioner

# Returns:
- Vector of interior vertices for BDDC preconditioner

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
p = 4
diffusion = GalleryOperator("diffusion", p, p, mesh);
bddc = LumpedBDDC(diffusion)

# note: either syntax works
interiorvertices = LFAToolkit.getinteriorvertices(bddc);
interiorvertices = bddc.interiorvertices;

# verify
trueinterfacevertices = [1:p..., p^2-p+1:p^2...]
for i = 1:p-2
    push!(trueinterfacevertices, i*p+1)
    push!(trueinterfacevertices, (i+1)*p)
end
trueinterfacevertices = sort(trueinterfacevertices)
@assert interiorvertices == setdiff(1:p^2, trueinterfacevertices)

# output

```
"""
function getinteriorvertices(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :interiorvertices)
        numbernodes, _ = size(bddc.fineoperator.elementmatrix)
        bddc.interiorvertices = setdiff(1:numbernodes, bddc.interfacevertices)
    end

    # return
    return getfield(bddc, :interiorvertices)
end

"""
```julia
getsubassembledinverse(bddc)
```

Compute or retrieve the subassembled subdomain solver for a BDDC preconditioner

# Returns:
- Matrix of inverse of subassembled subdomain matrix

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
p = 4
diffusion = GalleryOperator("diffusion", p, p, mesh);
bddc = LumpedBDDC(diffusion)

# note: either syntax works
subassembledinverse = LFAToolkit.getsubassembledinverse(bddc);
subassembledinverse = bddc.subassembledinverse;

# verify
@assert subassembledinverse ≈ Matrix(diffusion.elementmatrix[bddc.subassembledvertices, bddc.subassembledvertices])^-1

# output

```
"""
function getsubassembledinverse(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :subassembledinverse)
        elementmatrix = bddc.fineoperator.elementmatrix
        subassembledmatrix =
            Matrix(elementmatrix[bddc.subassembledvertices, bddc.subassembledvertices])
        bddc.subassembledinverse = subassembledmatrix^-1
    end

    # return
    return getfield(bddc, :subassembledinverse)
end

"""
```julia
getinteriorinverse(bddc)
```

Compute or retrieve the interior solver for a BDDC preconditioner

# Returns:
- Matrix of inverse of interior matrix

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
p = 4
diffusion = GalleryOperator("diffusion", p, p, mesh);
bddc = LumpedBDDC(diffusion)

# note: either syntax works
interiorinverse = LFAToolkit.getinteriorinverse(bddc);
interiorinverse = bddc.interiorinverse;

# verify
@assert interiorinverse ≈ Matrix(diffusion.elementmatrix[bddc.interiorvertices, bddc.interiorvertices])^-1

# output

```
"""
function getinteriorinverse(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :subassembledinverse)
        elementmatrix = bddc.fineoperator.elementmatrix
        interiormatrix = Matrix(elementmatrix[bddc.interiorvertices, bddc.interiorvertices])
        bddc.interiorinverse = interiormatrix^-1
    end

    # return
    return getfield(bddc, :interiorinverse)
end

# ------------------------------------------------------------------------------
# get/set property
# ------------------------------------------------------------------------------

function Base.getproperty(bddc::BDDC, f::Symbol)
    if f == :primalvertices
        return getprimalvertices(bddc)
    elseif f == :subassembledvertices
        return getsubassembledvertices(bddc)
    elseif f == :interfacevertices
        return getinterfacevertices(bddc)
    elseif f == :interiorvertices
        return getinteriorvertices(bddc)
    elseif f == :subassembledinverse
        return getsubassembledinverse(bddc)
    elseif f == :interiorinverse
        return getinteriorinverse(bddc)
    else
        return getfield(bddc, f)
    end
end

function Base.setproperty!(bddc::BDDC, f::Symbol, value)
    if f == :fineoperator || f == :injectiontype
        throw(ReadOnlyMemoryError()) # COV_EXCL_LINE
    else
        return setfield!(bddc, f, value)
    end
end

# ------------------------------------------------------------------------------
# compute symbols
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
