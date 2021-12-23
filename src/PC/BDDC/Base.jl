# ------------------------------------------------------------------------------
# bddc
# ------------------------------------------------------------------------------

"""
```julia
BDDC(operator, coarseoperator, smoother, prolongation)
```

BDDC preconditioner for finite element operators

# Arguments:
- `operator`:      finite element operator to precondition
- `injectiontype`: type of injection into subassembled space to use

# Returns:
- BDDC preconditioner object
"""
mutable struct BDDC <: AbstractPreconditioner
    # data never changed
    operator::Operator
    injectiontype::BDDCInjectionType.BDDCInjectType

    # data empty until assembled
    primalnodes::AbstractArray{Int,1}
    subassemblednodes::AbstractArray{Int,1}
    interfacenodes::AbstractArray{Int,1}
    interiornodes::AbstractArray{Int,1}
    primalmodes::AbstractArray{Int,1}
    subassembledmodes::AbstractArray{Int,1}
    interfacemodes::AbstractArray{Int,1}
    interiormodes::AbstractArray{Int,1}
    subassembledinverse::AbstractArray{Float64,2}
    interiorinverse::AbstractArray{Float64,2}
    schur::AbstractArray{Float64,2}
    mixedmultiplicity::AbstractArray{Float64}
    JDT::AbstractArray{Float64,2}
    primalrowmodemap::AbstractArray{Float64,2}
    primalcolumnmodemap::AbstractArray{Float64,2}
    subassembledrowmodemap::AbstractArray{Float64,2}
    subassembledcolumnmodemap::AbstractArray{Float64,2}
    mixedrowmodemap::AbstractArray{Float64,2}
    mixedcolumnmodemap::AbstractArray{Float64,2}
    modepermutation::AbstractArray{Bool,2}
    jacobi::Jacobi

    # inner constructor
    BDDC(operator::Operator, injectiontype::BDDCInjectionType.BDDCInjectType) = (
        # validity check
        if operator.dimension == 1
            error("BDDC preconditioner only valid for 1 and 2 dimensional operators") # COV_EXCL_LINE
        end;

        # constructor
        new(operator, injectiontype)
    )
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
getprimalnodes(bddc)
```

Compute or retrieve the primal nodes for a BDDC preconditioner

# Returns:
- Vector of primal nodes for BDDC preconditioner

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
p = 4
diffusion = GalleryOperator("diffusion", p, p, mesh);
bddc = LumpedBDDC(diffusion)

# note: either syntax works
primalnodes = LFAToolkit.getprimalnodes(bddc);
primalnodes = bddc.primalnodes;

# verify
@assert primalnodes == [1, p, p^2 - p + 1, p^2]

# output

```
"""
function getprimalnodes(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :primalnodes)
        operator = bddc.operator
        primalnodes = []
        numbernodes = 0
        for input in operator.inputs
            if input.evaluationmodes[1] != EvaluationMode.quadratureweights
                primalnodes =
                    vcat(primalnodes..., [input.basis.primalnodes .+ numbernodes]...)
                numbernodes += input.basis.numbernodes
            end
        end

        # store
        bddc.primalnodes = primalnodes
    end

    # return
    return getfield(bddc, :primalnodes)
end

"""
```julia
getsubassemblednodes(bddc)
```

Compute or retrieve the subassembled nodes for a BDDC preconditioner

# Returns:
- Vector of subassembled nodes for BDDC preconditioner

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
p = 4
diffusion = GalleryOperator("diffusion", p, p, mesh);
bddc = LumpedBDDC(diffusion)

# note: either syntax works
subassemblednodes = LFAToolkit.getsubassemblednodes(bddc);
subassemblednodes = bddc.subassemblednodes;

# verify
@assert subassemblednodes == setdiff(1:p^2, [1, p, p^2-p+1, p^2])

# output

```
"""
function getsubassemblednodes(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :subassemblednodes)
        numbernodes, _ = size(bddc.operator.elementmatrix)

        # store
        bddc.subassemblednodes = setdiff(1:numbernodes, bddc.primalnodes)
    end

    # return
    return getfield(bddc, :subassemblednodes)
end

"""
```julia
getinterfacenodes(bddc)
```

Compute or retrieve the interface nodes for a BDDC preconditioner

# Returns:
- Vector of interface nodes for BDDC preconditioner

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
p = 4
diffusion = GalleryOperator("diffusion", p, p, mesh);
bddc = LumpedBDDC(diffusion)

# note: either syntax works
interfacenodes = LFAToolkit.getinterfacenodes(bddc);
interfacenodes = bddc.interfacenodes;

# verify
trueinterfacenodes = [1:p..., p^2-p+1:p^2...]
for i = 1:p-2
    push!(trueinterfacenodes, i*p+1)
    push!(trueinterfacenodes, (i+1)*p)
end
trueinterfacenodes = sort(trueinterfacenodes)
trueinterfacenodes = setdiff(trueinterfacenodes, bddc.primalnodes)
@assert interfacenodes == trueinterfacenodes

# output

```
"""
function getinterfacenodes(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :interfacenodes)
        operator = bddc.operator
        interfacenodes = []
        numbernodes = 0
        for input in operator.inputs
            if input.evaluationmodes[1] != EvaluationMode.quadratureweights
                interfacenodes =
                    vcat(interfacenodes..., [input.basis.interfacenodes .+ numbernodes]...)
                numbernodes += input.basis.numbernodes
            end
        end
        interfacenodes = setdiff(interfacenodes, bddc.primalnodes)

        # store
        bddc.interfacenodes = interfacenodes
    end

    # return
    return getfield(bddc, :interfacenodes)
end

"""
```julia
getinteriornodes(bddc)
```

Compute or retrieve the interior nodes for a BDDC preconditioner

# Returns:
- Vector of interior nodes for BDDC preconditioner

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
p = 4
diffusion = GalleryOperator("diffusion", p, p, mesh);
bddc = LumpedBDDC(diffusion)

# note: either syntax works
interiornodes = LFAToolkit.getinteriornodes(bddc);
interiornodes = bddc.interiornodes;

# verify
trueinterfacenodes = [1:p..., p^2-p+1:p^2...]
for i = 1:p-2
    push!(trueinterfacenodes, i*p+1)
    push!(trueinterfacenodes, (i+1)*p)
end
trueinterfacenodes = sort(trueinterfacenodes)
trueinteriornodes = setdiff(bddc.subassemblednodes, trueinterfacenodes)
@assert interiornodes == trueinteriornodes

# output

```
"""
function getinteriornodes(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :interiornodes)
        # store
        bddc.interiornodes = setdiff(bddc.subassemblednodes, bddc.interfacenodes)
    end

    # return
    return getfield(bddc, :interiornodes)
end

"""
```julia
getprimalmodes(bddc)
```

Compute or retrieve the primal modes for a BDDC preconditioner

# Returns:
- Vector of primal modes for BDDC preconditioner

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
p = 4
diffusion = GalleryOperator("diffusion", p, p, mesh);
bddc = LumpedBDDC(diffusion)

# note: either syntax works
primalmodes = LFAToolkit.getprimalmodes(bddc);
primalmodes = bddc.primalmodes;

# verify
@assert primalmodes == [1]

# output

```
"""
function getprimalmodes(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :primalmodes)
        primalmodes::Array{Int,1} = []
        numberprimalnodes = maximum(size(bddc.primalnodes))
        numbermodes, _ = size(bddc.operator.rowmodemap)
        modemap = bddc.operator.rowmodemap[:, bddc.primalnodes] * ones(numberprimalnodes)
        for i = 1:numbermodes
            if modemap[i] > 0.0
                push!(primalmodes, i)
            end
        end

        # store
        bddc.primalmodes = primalmodes
    end

    # return
    return getfield(bddc, :primalmodes)
end

"""
```julia
getsubassembledmodes(bddc)
```

Compute or retrieve the subassembled modes for a BDDC preconditioner

# Returns:
- Vector of subassembled modes for BDDC preconditioner

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
p = 4
diffusion = GalleryOperator("diffusion", p, p, mesh);
bddc = LumpedBDDC(diffusion)

# note: either syntax works
subassembledmodes = LFAToolkit.getsubassembledmodes(bddc);
subassembledmodes = bddc.subassembledmodes;

# verify
@assert subassembledmodes == setdiff(1:(p-1)^2, [1])

# output

```
"""
function getsubassembledmodes(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :subassembledmodes)
        numbermodes, _ = size(bddc.operator.rowmodemap)

        # store
        bddc.subassembledmodes = setdiff(1:numbermodes, bddc.primalmodes)
    end

    # return
    return getfield(bddc, :subassembledmodes)
end

"""
```julia
getinterfacemodes(bddc)
```

Compute or retrieve the interface modes for a BDDC preconditioner

# Returns:
- Vector of interface modes for BDDC preconditioner

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
p = 4
diffusion = GalleryOperator("diffusion", p, p, mesh);
bddc = LumpedBDDC(diffusion)

# note: either syntax works
interfacemodes = LFAToolkit.getinterfacemodes(bddc);
interfacemodes = bddc.interfacemodes;

# verify
trueinterfacemodes = [1:p-1...]
for i = 1:p-2
    push!(trueinterfacemodes, i*(p-1)+1)
end
trueinterfacemodes = sort(trueinterfacemodes)
trueinterfacemodes = setdiff(trueinterfacemodes, bddc.primalmodes)
@assert interfacemodes == trueinterfacemodes

# output

```
"""
function getinterfacemodes(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :interfacemodes)
        interfacemodes::Array{Int,1} = []
        numberinterfacenodes = maximum(size(bddc.interfacenodes))
        numbermodes, _ = size(bddc.operator.rowmodemap)
        modemap =
            bddc.operator.rowmodemap[:, bddc.interfacenodes] * ones(numberinterfacenodes)
        for i = 1:numbermodes
            if modemap[i] > 0.0
                push!(interfacemodes, i)
            end
        end
        interfacemodes = setdiff(interfacemodes, bddc.primalmodes)

        # store
        bddc.interfacemodes = interfacemodes
    end

    # return
    return getfield(bddc, :interfacemodes)
end

"""
```julia
getinteriormodes(bddc)
```

Compute or retrieve the interior modes for a BDDC preconditioner

# Returns:
- Vector of interior modes for BDDC preconditioner

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
p = 4
diffusion = GalleryOperator("diffusion", p, p, mesh);
bddc = LumpedBDDC(diffusion)

# note: either syntax works
interiormodes = LFAToolkit.getinteriormodes(bddc);
interiormodes = bddc.interiormodes;

# verify
@assert interiormodes == setdiff(bddc.subassembledmodes, bddc.interfacemodes)

# output

```
"""
function getinteriormodes(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :interiormodes)
        # store
        bddc.interiormodes = setdiff(bddc.subassembledmodes, bddc.interfacemodes)
    end

    # return
    return getfield(bddc, :interiormodes)
end

"""
```julia
getsubassembledinverse(bddc)
```

Compute or retrieve the solver for subdomain for a BDDC preconditioner

# Returns:
- Solver for subdomain

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
@assert subassembledinverse ≈ Matrix(diffusion.elementmatrix[bddc.subassemblednodes, bddc.subassemblednodes])^-1

# output

```
"""
function getsubassembledinverse(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :subassembledinverse)
        elementmatrix = bddc.operator.elementmatrix
        subassembledmatrix =
            Matrix(elementmatrix[bddc.subassemblednodes, bddc.subassemblednodes])

        # store
        bddc.subassembledinverse = subassembledmatrix^-1
    end

    # return
    return getfield(bddc, :subassembledinverse)
end

"""
```julia
getinteriorinverse(bddc)
```

Compute or retrieve the solver for subdomain interior for a BDDC preconditioner

# Returns:
- Solver for subdomain interior

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
@assert interiorinverse ≈ Matrix(diffusion.elementmatrix[bddc.interiornodes, bddc.interiornodes])^-1

# output

```
"""
function getinteriorinverse(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :interiorinverse)
        elementmatrix = bddc.operator.elementmatrix
        interiormatrix = Matrix(elementmatrix[bddc.interiornodes, bddc.interiornodes])

        # store
        bddc.interiorinverse = interiormatrix^-1
    end

    # return
    return getfield(bddc, :interiorinverse)
end

"""
```julia
getschur(bddc)
```

Compute or retrieve the Schur complement matrix for a BDDC preconditioner

# Returns:
- Matrix for Schur complement

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
p = 4
diffusion = GalleryOperator("diffusion", p, p, mesh);
bddc = LumpedBDDC(diffusion)

# note: either syntax works
schur = LFAToolkit.getschur(bddc);
schur = bddc.schur;

# verify
@assert schur ≈ diffusion.elementmatrix[bddc.primalnodes, bddc.primalnodes] -
    diffusion.elementmatrix[bddc.primalnodes, bddc.subassemblednodes] *
    Matrix(diffusion.elementmatrix[bddc.subassemblednodes, bddc.subassemblednodes])^-1 *
    diffusion.elementmatrix[bddc.subassemblednodes, bddc.primalnodes]

# output

```
"""
function getschur(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :schur)
        elementmatrix = bddc.operator.elementmatrix
        mixedmatrix = elementmatrix[bddc.primalnodes, bddc.subassemblednodes]
        schur =
            elementmatrix[bddc.primalnodes, bddc.primalnodes] -
            mixedmatrix * bddc.subassembledinverse * mixedmatrix'

        # store
        bddc.schur = schur
    end

    # return
    return getfield(bddc, :schur)
end

"""
```julia
getprimalrowmodemap(bddc)
```

Compute or retrieve the matrix mapping the rows of the primal BDDC matrix to the primal symbol matrix

# Returns:
- Matrix mapping rows of primal BDDC matrix to primal symbol matrix

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
diffusion = GalleryOperator("diffusion", 3, 3, mesh);
bddc = LumpedBDDC(diffusion);

# note: either syntax works
primalmodemap = LFAToolkit.getprimalrowmodemap(bddc);
primalmodemap = bddc.primalrowmodemap;

# verify
@assert primalmodemap ≈ [1 1 1 1]

# output

```
"""
function getprimalrowmodemap(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :primalrowmodemap)
        primalrowmodemap = bddc.operator.rowmodemap[bddc.primalmodes, bddc.primalnodes]

        # store
        bddc.primalrowmodemap = primalrowmodemap
    end

    # return
    return getfield(bddc, :primalrowmodemap)
end

"""
```julia
getprimalcolumnmodemap(bddc)
```

Compute or retrieve the matrix mapping the columns of the primal BDDC matrix to the
primal symbol matrix

# Returns:
- Matrix mapping columns of primal BDDC matrix to primal symbol matrix

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
diffusion = GalleryOperator("diffusion", 3, 3, mesh);
bddc = LumpedBDDC(diffusion);

# note: either syntax works
primalmodemap = LFAToolkit.getprimalcolumnmodemap(bddc);
primalmodemap = bddc.primalcolumnmodemap;

# verify
@assert primalmodemap ≈ [1; 1; 1; 1]

# output

```
"""
function getprimalcolumnmodemap(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :primalcolumnmodemap)
        primalcolumnmodemap =
            bddc.operator.columnmodemap[bddc.primalnodes, bddc.primalmodes]

        # store
        bddc.primalcolumnmodemap = primalcolumnmodemap
    end

    # return
    return getfield(bddc, :primalcolumnmodemap)
end

"""
```julia
getsubassembledrowmodemap(bddc)
```

Compute or retrieve the matrix mapping the rows of the subassembled BDDC matrix to the
subassembled symbol matrix

# Returns:
- Matrix mapping rows of subassembled BDDC matrix to subassembled symbol matrix

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
diffusion = GalleryOperator("diffusion", 3, 3, mesh);
bddc = LumpedBDDC(diffusion);

# note: either syntax works
subassembledmodemap = LFAToolkit.getsubassembledrowmodemap(bddc);
subassembledmodemap = bddc.subassembledrowmodemap;

# verify
@assert subassembledmodemap ≈ bddc.operator.rowmodemap[bddc.subassembledmodes, bddc.subassemblednodes]

# output

```
"""
function getsubassembledrowmodemap(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :subassembledrowmodemap)
        subassembledrowmodemap =
            bddc.operator.rowmodemap[bddc.subassembledmodes, bddc.subassemblednodes]

        # store
        bddc.subassembledrowmodemap = subassembledrowmodemap
    end

    # return
    return getfield(bddc, :subassembledrowmodemap)
end

"""
```julia
getsubassembledcolumnmodemap(bddc)
```

Compute or retrieve the matrix mapping the columns of the subassembled BDDC matrix to the
subassembled symbol matrix

# Returns:
- Matrix mapping columns of subassembled BDDC matrix to subassembled symbol matrix

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
diffusion = GalleryOperator("diffusion", 3, 3, mesh);
bddc = LumpedBDDC(diffusion);

# note: either syntax works
subassembledmodemap = LFAToolkit.getsubassembledcolumnmodemap(bddc);
subassembledmodemap = bddc.subassembledcolumnmodemap;

# verify
@assert subassembledmodemap ≈ bddc.operator.columnmodemap[bddc.subassemblednodes, bddc.subassembledmodes]

# output

```
"""
function getsubassembledcolumnmodemap(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :subassembledcolumnmodemap)
        subassembledcolumnmodemap =
            bddc.operator.columnmodemap[bddc.subassemblednodes, bddc.subassembledmodes]

        # store
        bddc.subassembledcolumnmodemap = subassembledcolumnmodemap
    end

    # return
    return getfield(bddc, :subassembledcolumnmodemap)
end

"""
```julia
getmixedrowmodemap(bddc)
```

Compute or retrieve the matrix mapping the rows of the mixed BDDC matrix to the
symbol matrix

# Returns:
- Matrix mapping rows of mixed BDDC matrix to symbol matrix
"""
function getmixedrowmodemap(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :mixedrowmodemap)
        numberprimalmodes = maximum(size(bddc.primalmodes))
        numbersubassembledmodes = maximum(size(bddc.subassembledmodes))
        numbersubassemblednodes = maximum(size(bddc.subassemblednodes))
        mixedrowmodemap = [
            I(numberprimalmodes) zeros((numberprimalmodes, numbersubassemblednodes))
            zeros(numbersubassembledmodes, numberprimalmodes) bddc.subassembledrowmodemap
        ]

        # store
        bddc.mixedrowmodemap = bddc.modepermutation * mixedrowmodemap
    end

    # return
    return getfield(bddc, :mixedrowmodemap)
end

"""
```julia
getmixedcolumnmodemap(bddc)
```

Compute or retrieve the matrix mapping the columns of the mixed BDDC matrix to the
symbol matrix

# Returns:
- Matrix mapping columns of mixed BDDC matrix to symbol matrix
"""
function getmixedcolumnmodemap(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :mixedcolumnmodemap)
        numberprimalmodes = maximum(size(bddc.primalmodes))
        numbersubassembledmodes = maximum(size(bddc.subassembledmodes))
        numbersubassemblednodes = maximum(size(bddc.subassemblednodes))
        mixedcolumnmodemap = [
            I(numberprimalmodes) zeros((numberprimalmodes, numbersubassembledmodes))
            zeros(numbersubassemblednodes, numberprimalmodes) bddc.subassembledcolumnmodemap
        ]

        # store
        bddc.mixedcolumnmodemap = mixedcolumnmodemap
    end

    # return
    return getfield(bddc, :mixedcolumnmodemap)

end

"""
```julia
getmodepermutation(bddc)
```

Compute or retrieve the matrix permuting multi-component modes to standard ordering

# Returns:
- Matrix mapping BDDC mode ordering to standard ordering
"""
function getmodepermutation(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :modepermutation)
        numberprimal = maximum(size(bddc.primalmodes))
        numbersubassembled = maximum(size(bddc.subassembledmodes))
        numbermodes = numberprimal + numbersubassembled
        modepermutation = spzeros(Bool, numbermodes, numbermodes)

        # build permutation matrix
        for i = 1:numberprimal
            modepermutation[bddc.primalmodes[i], i] = true
        end
        for i = 1:numbersubassembled
            modepermutation[bddc.subassembledmodes[i], i+numberprimal] = true
        end

        # store
        bddc.modepermutation = modepermutation
    end

    # return
    return getfield(bddc, :modepermutation)
end

"""
```julia
getmixedmultiplicity(bddc)
```

Compute or retrieve the diagonal matrix of mixed interface node and primal mode
multiplicity for the BDDC preconditioner

# Returns:
- Matrix of mixed multiplicity for the BDDC preconditioner
"""
function getmixedmultiplicity(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :mixedmultiplicity)
        numberprimalmodes = maximum(size(bddc.primalmodes))
        mixedmultiplicity = Diagonal(
            vcat(
                ones(numberprimalmodes, 1)...,
                [1 / m for m in bddc.operator.multiplicity[bddc.subassemblednodes]]...,
            ),
        )

        # store
        bddc.mixedmultiplicity = mixedmultiplicity
    end

    # return
    return getfield(bddc, :mixedmultiplicity)
end

"""
```julia
getJDT(bddc)
```

Compute or retrieve the matrix mapping the jump over subdomain interfacemodes

# Returns:
- Matrix mapping the jump over subdomain interfacemodes
"""
function getJDT(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :JDT)
        # setup
        numberprimalmodes = maximum(size(bddc.primalmodes))
        numbernodes, _ = size(bddc.operator.columnmodemap)
        numbermixed, _ = size(bddc.mixedcolumnmodemap)
        numberinterfacenodes = maximum(size(bddc.interfacenodes))

        # build jump map
        J_D_T_nodes = zeros(numbernodes, numbernodes)
        for i = 1:numberinterfacenodes, j = 1:numberinterfacenodes
            indxi = bddc.interfacenodes[i]
            indxj = bddc.interfacenodes[j]
            if bddc.operator.rowmodemap[:, indxi] == bddc.operator.rowmodemap[:, indxj]
                scale = -1 / bddc.operator.multiplicity[indxi]
                if i == j
                    scale = 1 + scale
                end
                J_D_T_nodes[indxi, indxj] = scale
            end
        end
        J_D_T_mixed = spzeros(numbermixed, numbermixed)
        J_D_T_mixed[numberprimalmodes+1:end, numberprimalmodes+1:end] =
            J_D_T_nodes[bddc.subassemblednodes, bddc.subassemblednodes]

        # store
        bddc.JDT = J_D_T_mixed
    end

    # return
    return getfield(bddc, :JDT)
end

# ------------------------------------------------------------------------------
# data for computing symbols
# ------------------------------------------------------------------------------

"""
```julia
computesymbolsinjection(bddc)
```

Compute or retrieve the injection operator for the BDDC symbol matrix

# Returns:
- Matrix providing the injection operator of BDDC symbol matrix

"""
function computesymbolsinjection(bddc::BDDC, θ::Array)
    if (bddc.injectiontype == BDDCInjectionType.scaled)
        # lumped BDDC
        return bddc.mixedmultiplicity
    elseif (bddc.injectiontype == BDDCInjectionType.harmonic)
        # Dirichlet BDDC
        # -- validity check
        dimension = length(θ)
        if dimension != bddc.operator.inputs[1].basis.dimension
            # COV_EXCL_START
            throw(
                ArgumentError(
                    "Must provide as many values of θ as the mesh has dimensions",
                ),
            )
            # COV_EXCL_STOP
        end

        # -- setup
        numberprimalmodes = maximum(size(bddc.primalmodes))
        numbernodes, _ = size(bddc.operator.columnmodemap)
        numbermixed, _ = size(bddc.mixedcolumnmodemap)
        numberinteriornodes = maximum(size(bddc.interiornodes))
        numberinterfacenodes = maximum(size(bddc.interfacenodes))
        nodecoordinatedifferences = bddc.operator.nodecoordinatedifferences
        elementmatrix = bddc.operator.elementmatrix

        # -- harmonic operator
        A_ΓI_nodes = zeros(ComplexF64, numbernodes, numbernodes)
        for i = 1:numberinterfacenodes, j = 1:numberinteriornodes
            indxi = bddc.interfacenodes[i]
            indxj = bddc.interiornodes[j]
            A_ΓI_nodes[indxi, indxj] =
                elementmatrix[indxi, indxj] *
                ℯ^(
                    im * sum([
                        θ[k] * nodecoordinatedifferences[indxi, indxj, k] for
                        k = 1:dimension
                    ])
                )
        end
        A_II_inv = bddc.interiorinverse
        A_II_inv_nodes = zeros(ComplexF64, numbernodes, numbernodes)
        for i = 1:numberinteriornodes, j = 1:numberinteriornodes
            indxi = bddc.interiornodes[i]
            indxj = bddc.interiornodes[j]
            A_II_inv_nodes[indxi, indxj] =
                A_II_inv[i, j] *
                ℯ^(
                    im * sum([
                        θ[k] * nodecoordinatedifferences[indxi, indxj, k] for
                        k = 1:dimension
                    ])
                )
        end
        𝓗_T = A_ΓI_nodes * A_II_inv_nodes
        𝓗_T_mixed = spzeros(ComplexF64, numbermixed, numbermixed)
        𝓗_T_mixed[numberprimalmodes+1:end, numberprimalmodes+1:end] =
            𝓗_T[bddc.subassemblednodes, bddc.subassemblednodes]

        # -- jump mapping
        J_D_T_mixed = bddc.JDT

        # -- injection mapping
        return bddc.mixedmultiplicity + J_D_T_mixed * 𝓗_T_mixed
    else
        throw(ArgumentError("Injection type unknown")) # COV_EXCL_LINE
    end
end

# ------------------------------------------------------------------------------
# get/set property
# ------------------------------------------------------------------------------

function Base.getproperty(bddc::BDDC, f::Symbol)
    if f == :primalnodes
        return getprimalnodes(bddc)
    elseif f == :subassemblednodes
        return getsubassemblednodes(bddc)
    elseif f == :interfacenodes
        return getinterfacenodes(bddc)
    elseif f == :interiornodes
        return getinteriornodes(bddc)
    elseif f == :primalmodes
        return getprimalmodes(bddc)
    elseif f == :subassembledmodes
        return getsubassembledmodes(bddc)
    elseif f == :interfacemodes
        return getinterfacemodes(bddc)
    elseif f == :interiormodes
        return getinteriormodes(bddc)
    elseif f == :subassembledinverse
        return getsubassembledinverse(bddc)
    elseif f == :interiorinverse
        return getinteriorinverse(bddc)
    elseif f == :schur
        return getschur(bddc)
    elseif f == :mixedmultiplicity
        return getmixedmultiplicity(bddc)
    elseif f == :JDT
        return getJDT(bddc)
    elseif f == :primalrowmodemap
        return getprimalrowmodemap(bddc)
    elseif f == :primalcolumnmodemap
        return getprimalcolumnmodemap(bddc)
    elseif f == :subassembledrowmodemap
        return getsubassembledrowmodemap(bddc)
    elseif f == :subassembledcolumnmodemap
        return getsubassembledcolumnmodemap(bddc)
    elseif f == :mixedrowmodemap
        return getmixedrowmodemap(bddc)
    elseif f == :mixedcolumnmodemap
        return getmixedcolumnmodemap(bddc)
    elseif f == :modepermutation
        return getmodepermutation(bddc)
    else
        return getfield(bddc, f)
    end
end

function Base.setproperty!(bddc::BDDC, f::Symbol, value)
    if f == :operator || f == :injectiontype
        throw(ReadOnlyMemoryError()) # COV_EXCL_LINE
    else
        return setfield!(bddc, f, value)
    end
end

# ------------------------------------------------------------------------------
# compute symbols
# ------------------------------------------------------------------------------

"""
```julia
computesymbols(bddc, θ)
```

Compute the symbol matrix for a BDDC preconditioned operator

# Arguments:
- `bddc`: BDDC preconditioner to compute symbol matrix for
- `θ`:    Fourier mode frequency array (one frequency per dimension)

# Returns:
- Symbol matrix for the BDDC preconditioned operator

# Lumped BDDC Example:
```jldoctest
using LinearAlgebra;

for dimension in 2:3
    # setup
    mesh = []
    if dimension == 2
        mesh = Mesh2D(1.0, 1.0);
    elseif dimension == 3
        mesh = Mesh3D(1.0, 1.0, 1.0);
    end
    diffusion = GalleryOperator("diffusion", 3, 3, mesh);
    bddc = LumpedBDDC(diffusion)

    # compute symbols
    A = computesymbols(bddc, [0.2], π*ones(dimension));

    # verify
    eigenvalues = real(eigvals(A));
    if dimension == 2
        @assert minimum(eigenvalues) ≈ 0.43999999999999995
        @assert maximum(eigenvalues) ≈ 0.8
    elseif dimension == 3
        @assert minimum(eigenvalues) ≈ -0.6319999999999972
        @assert maximum(eigenvalues) ≈ 0.8
    end
end

# output

```

# Dirichlet BDDC Example:
```jldoctest
using LinearAlgebra;

for dimension in 2:3
    # setup
    mesh = []
    if dimension == 2
        mesh = Mesh2D(1.0, 1.0);
    elseif dimension == 3
        mesh = Mesh3D(1.0, 1.0, 1.0);
    end
    diffusion = GalleryOperator("diffusion", 3, 3, mesh);
    bddc = DirichletBDDC(diffusion)

    # compute symbols
    A = computesymbols(bddc, [0.2], π*ones(dimension));

    # verify
    eigenvalues = real(eigvals(A));
    if dimension == 2
        @assert minimum(eigenvalues) ≈ 0.7999999999999998
        @assert maximum(eigenvalues) ≈ 0.8
    elseif dimension == 3
        @assert minimum(eigenvalues) ≈ 0.7801226993865031
        @assert maximum(eigenvalues) ≈ 0.8
    end
end

# output

```
"""
function computesymbols(bddc::BDDC, ω::Array, θ::Array)
    # validity check
    if length(ω) != 1
        Throw(error("exactly one parameter required for BDDC smoothing")) # COV_EXCL_LINE
    end
    dimension = length(θ)
    if dimension != bddc.operator.inputs[1].basis.dimension
        throw(ArgumentError("Must provide as many values of θ as the mesh has dimensions")) # COV_EXCL_LINE
    end

    # setup
    elementmatrix = bddc.operator.elementmatrix
    numberprimalnodes = maximum(size(bddc.primalnodes))
    numberprimalmodes = maximum(size(bddc.primalmodes))
    numbersubassemblednodes = maximum(size(bddc.subassemblednodes))
    nodecoordinatedifferences = bddc.operator.nodecoordinatedifferences

    # subdomain solver
    A_rr_inv = bddc.subassembledinverse
    A_rr_inv_nodes = zeros(ComplexF64, numbersubassemblednodes, numbersubassemblednodes)
    for i = 1:numbersubassemblednodes, j = 1:numbersubassemblednodes
        indxi = bddc.subassemblednodes[i]
        indxj = bddc.subassemblednodes[j]
        A_rr_inv_nodes[i, j] =
            A_rr_inv[i, j] *
            ℯ^(
                im * sum([
                    θ[k] * nodecoordinatedifferences[indxi, indxj, k] for k = 1:dimension
                ])
            )
    end

    # mixed subassembled primal matrices
    Â_Πr_nodes = zeros(ComplexF64, numberprimalnodes, numbersubassemblednodes)
    for i = 1:numberprimalnodes, j = 1:numbersubassemblednodes
        indxi = bddc.primalnodes[i]
        indxj = bddc.subassemblednodes[j]
        Â_Πr_nodes[i, j] =
            elementmatrix[indxi, indxj] *
            ℯ^(
                im * sum([
                    θ[k] * nodecoordinatedifferences[indxi, indxj, k] for k = 1:dimension
                ])
            )
    end
    Â_Πr_modes = bddc.primalrowmodemap * Â_Πr_nodes

    # Schur complement
    Ŝ_Π = bddc.schur
    Ŝ_Π_nodes = zeros(ComplexF64, numberprimalnodes, numberprimalnodes)
    for i = 1:numberprimalnodes, j = 1:numberprimalnodes
        indxi = bddc.primalnodes[i]
        indxj = bddc.primalnodes[j]
        Ŝ_Π_nodes[i, j] =
            Ŝ_Π[i, j] *
            ℯ^(
                im * sum([
                    θ[k] * nodecoordinatedifferences[indxi, indxj, k] for k = 1:dimension
                ])
            )
    end
    Ŝ_Π_inv_modes = (bddc.primalrowmodemap * Ŝ_Π_nodes * bddc.primalcolumnmodemap)^-1

    # subassembled nodes primal modes
    Ø = zeros((numberprimalmodes, numbersubassemblednodes))
    K_u_inv = [
        I(numberprimalmodes) Ø
        -A_rr_inv_nodes*Â_Πr_modes' I(numbersubassemblednodes)
    ]
    P_inv = [
        Ŝ_Π_inv_modes Ø
        Ø' A_rr_inv_nodes
    ]
    Â_inv_mixed = K_u_inv * P_inv * K_u_inv'

    # injection
    R_mixed = computesymbolsinjection(bddc, θ)
    R_T_Â_inv_R_modes =
        bddc.mixedrowmodemap * R_mixed' * Â_inv_mixed * R_mixed * bddc.mixedcolumnmodemap

    # return
    return I - ω[1] * R_T_Â_inv_R_modes * computesymbols(bddc.operator, θ)
end

# ------------------------------------------------------------------------------
