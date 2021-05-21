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
    primalnodes::AbstractArray{Int,1}
    subassemblednodes::AbstractArray{Int,1}
    interfacenodes::AbstractArray{Int,1}
    interiornodes::AbstractArray{Int,1}
    primalmodes::AbstractArray{Int,1}
    subassembledmodes::AbstractArray{Int,1}
    subassembledinverse::AbstractArray{Float64,2}
    interiorinverse::AbstractArray{Float64,2}
    schur::AbstractArray{Float64,2}
    injection::AbstractArray{Float64}
    rowmodemap::AbstractArray{Float64,2}
    columnmodemap::AbstractArray{Float64,2}
    primalrowmodemap::AbstractArray{Float64,2}
    primalcolumnmodemap::AbstractArray{Float64,2}

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
getprimalnodes(bddc)
```

Compute or retrieve the primal nodes for a BDDC preconditioner

# Returns:
- Vector of primal nodes for BDDC preconditioner

# Example:
```jldoctest
# setup
mesh = Mesh1D(1.0);
p = 4
diffusion = GalleryOperator("diffusion", p, p, mesh);
bddc = LumpedBDDC(diffusion)

# note: either syntax works
primalnodes = LFAToolkit.getprimalnodes(bddc);
primalnodes = bddc.primalnodes;

# verify
@assert primalnodes == [1, p]

# output

```
"""
function getprimalnodes(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :primalnodes)
        operator = bddc.fineoperator
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
        numbernodes, _ = size(bddc.fineoperator.elementmatrix)

        # store
        bddc.subassemblednodes = setdiff(1:numbernodes, bddc.primalnodes)
    end

    # return
    return getfield(bddc, :subassemblednodes)
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
mesh = Mesh1D(1.0);
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
        numberprimalnodes = max(size(bddc.primalnodes)...)
        numbermodes, _ = size(bddc.fineoperator.rowmodemap)
        modemap =
            bddc.fineoperator.rowmodemap[:, bddc.primalnodes] * ones(numberprimalnodes)
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
        numbermodes, _ = size(bddc.fineoperator.rowmodemap)

        # store
        bddc.subassembledmodes = setdiff(1:numbermodes, bddc.primalmodes)
    end

    # return
    return getfield(bddc, :subassembledmodes)
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
mesh = Mesh1D(1.0);
p = 4
diffusion = GalleryOperator("diffusion", p, p, mesh);
bddc = LumpedBDDC(diffusion)

# note: either syntax works
interfacenodes = LFAToolkit.getinterfacenodes(bddc);
interfacenodes = bddc.interfacenodes;

# verify
@assert interfacenodes == [1, p]

# output

```
"""
function getinterfacenodes(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :interfacenodes)
        operator = bddc.fineoperator
        interfacenodes = []
        numbernodes = 0
        for input in operator.inputs
            if input.evaluationmodes[1] != EvaluationMode.quadratureweights
                interfacenodes =
                    vcat(interfacenodes..., [input.basis.interfacenodes .+ numbernodes]...)
                numbernodes += input.basis.numbernodes
            end
        end

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
@assert interiornodes == setdiff(1:p^2, trueinterfacenodes)

# output

```
"""
function getinteriornodes(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :interiornodes)
        numbernodes, _ = size(bddc.fineoperator.elementmatrix)

        # store
        bddc.interiornodes = setdiff(1:numbernodes, bddc.interfacenodes)
    end

    # return
    return getfield(bddc, :interiornodes)
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
        elementmatrix = bddc.fineoperator.elementmatrix
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
        elementmatrix = bddc.fineoperator.elementmatrix
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
        elementmatrix = bddc.fineoperator.elementmatrix
        schur =
            elementmatrix[bddc.primalnodes, bddc.primalnodes] -
            elementmatrix[bddc.primalnodes, bddc.subassemblednodes] *
            bddc.subassembledinverse *
            elementmatrix[bddc.subassemblednodes, bddc.primalnodes]

        # store
        bddc.schur = schur
    end

    # return
    return getfield(bddc, :schur)
end

"""
```julia
getcolumnmodemap(bddc)
```

Compute or retrieve the injection operator for the mixed BDDC matrix

# Returns:
- Matrix providing the injection operator of mixed BDDC matrix

# Example:
"""
function getinjection(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :injection)
        numberprimalmodes = max(size(bddc.primalmodes)...)
        injection = []
        if (bddc.injectiontype == BDDCInjectionType.scaled)
            # lumped BDDC
            injection = Diagonal(
                vcat(
                    bddc.fineoperator.multiplicity[bddc.subassemblednodes] .^ -1...,
                    ones(numberprimalmodes, 1)...,
                ),
            )
        elseif (bddc.injectiontype == BDDCInjectionType.harmonic)
            # Dirichlet BDDC
            throw(ArgumentError("Injection type not yet supported")) # COV_EXCL_LINE
        else
            throw(ArgumentError("Injection type unknown")) # COV_EXCL_LINE
        end

        # store
        bddc.injection = injection
    end

    # return
    return getfield(bddc, :injection)
end

"""
```julia
getrowmodemap(bddc)
```

Compute or retrieve the matrix mapping the rows of the mixed BDDC matrix to the symbol matrix

# Returns:
- Matrix mapping rows of mixed BDDC matrix to symbol matrix

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
finediffusion = GalleryOperator("diffusion", 3, 3, mesh);
bddc = LumpedBDDC(finediffusion);

# note: either syntax works
modemap = LFAToolkit.getrowmodemap(bddc);
modemap = bddc.rowmodemap;

# verify
@assert modemap ≈ [1 0 0 0 1 0; 0 1 0 1 0 0; 0 0 1 0 0 0; 0 0 0 0 0 1]

# output

```
"""
function getrowmodemap(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :rowmodemap)
        numbermodes, _ = size(bddc.fineoperator.rowmodemap)
        numberprimalmodes = max(size(bddc.primalmodes)...)
        numbermixed = numberprimalmodes + max(size(bddc.subassemblednodes)...)
        rowmodemap = spzeros(numbermodes, numbermixed)
        rowmodemap[1:end-numberprimalmodes, 1:end-numberprimalmodes] =
            bddc.fineoperator.rowmodemap[bddc.subassembledmodes, bddc.subassemblednodes]
        for i = 0:numberprimalmodes-1
            rowmodemap[numbermodes-i, numbermixed-i] = 1
        end

        # store
        bddc.rowmodemap = rowmodemap
    end

    # return
    return getfield(bddc, :rowmodemap)
end

"""
```julia
getcolumnmodemap(bddc)
```

Compute or retrieve the matrix mapping the columns of the mixed BDDC matrix to the
  symbol matrix

# Returns:
- Matrix mapping columns of mixed BDDC matrix to symbol matrix

# Example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
finediffusion = GalleryOperator("diffusion", 3, 3, mesh);
bddc = LumpedBDDC(finediffusion);

# note: either syntax works
modemap = LFAToolkit.getcolumnmodemap(bddc);
modemap = bddc.columnmodemap;

# verify
@assert modemap ≈ [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 1 0 0; 1 0 0 0; 0 0 0 1]

# output

```
"""
function getcolumnmodemap(bddc::BDDC)
    # assemble if needed
    if !isdefined(bddc, :columnmodemap)
        _, numbermodes = size(bddc.fineoperator.columnmodemap)
        numberprimalmodes = max(size(bddc.primalmodes)...)
        numbermixed = numberprimalmodes + max(size(bddc.subassemblednodes)...)
        columnmodemap = spzeros(numbermixed, numbermodes)
        columnmodemap[1:end-numberprimalmodes, 1:end-numberprimalmodes] =
            bddc.fineoperator.columnmodemap[bddc.subassemblednodes, bddc.subassembledmodes]
        for i = 0:numberprimalmodes-1
            columnmodemap[numbermixed-i, numbermodes-i] = 1
        end

        # store
        bddc.columnmodemap = columnmodemap
    end

    # return
    return getfield(bddc, :columnmodemap)
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
finediffusion = GalleryOperator("diffusion", 3, 3, mesh);
bddc = LumpedBDDC(finediffusion);

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
    if !isdefined(bddc, :rowmodemap)
        primalrowmodemap = bddc.fineoperator.rowmodemap[bddc.primalmodes, bddc.primalnodes]

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
finediffusion = GalleryOperator("diffusion", 3, 3, mesh);
bddc = LumpedBDDC(finediffusion);

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
            bddc.fineoperator.columnmodemap[bddc.primalnodes, bddc.primalmodes]

        # store
        bddc.primalcolumnmodemap = primalcolumnmodemap
    end

    # return
    return getfield(bddc, :primalcolumnmodemap)
end

# ------------------------------------------------------------------------------
# get/set property
# ------------------------------------------------------------------------------

function Base.getproperty(bddc::BDDC, f::Symbol)
    if f == :primalnodes
        return getprimalnodes(bddc)
    elseif f == :subassemblednodes
        return getsubassemblednodes(bddc)
    elseif f == :primalmodes
        return getprimalmodes(bddc)
    elseif f == :subassembledmodes
        return getsubassembledmodes(bddc)
    elseif f == :interfacenodes
        return getinterfacenodes(bddc)
    elseif f == :interiornodes
        return getinteriornodes(bddc)
    elseif f == :subassembledinverse
        return getsubassembledinverse(bddc)
    elseif f == :interiorinverse
        return getinteriorinverse(bddc)
    elseif f == :schur
        return getschur(bddc)
    elseif f == :injection
        return getinjection(bddc)
    elseif f == :rowmodemap
        return getrowmodemap(bddc)
    elseif f == :columnmodemap
        return getcolumnmodemap(bddc)
    elseif f == :primalrowmodemap
        return getprimalrowmodemap(bddc)
    elseif f == :primalcolumnmodemap
        return getprimalcolumnmodemap(bddc)
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
    bddc = LumpedBDDC(diffusion)

    # compute symbols
    A = computesymbols(bddc, π*ones(dimension));

    # verify

end

# output

```
"""
function computesymbols(bddc::BDDC, θ::Array)
    # validity check
    dimension = length(θ)
    if dimension != bddc.fineoperator.inputs[1].basis.dimension
        throw(ArgumentError("Must provide as many values of θ as the mesh has dimensions")) # COV_EXCL_LINE
    end

    # setup
    elementmatrix = bddc.fineoperator.elementmatrix
    numberprimalnodes = max(size(bddc.primalnodes)...)
    numberprimalmodes = max(size(bddc.primalmodes)...)
    numbersubassemblednodes = max(size(bddc.subassemblednodes)...)
    nodecoordinatedifferences = bddc.fineoperator.nodecoordinatedifferences

    # subdomain solver
    A_rr_inv = bddc.subassembledinverse
    A_rr_inv_nodes = zeros(ComplexF64, numbersubassemblednodes, numbersubassemblednodes)
    if dimension == 1
        for i = 1:numbersubassemblednodes, j = 1:numbersubassemblednodes
            indxi = bddc.subassemblednodes[i]
            indxj = bddc.subassemblednodes[j]
            A_rr_inv_nodes[i, j] =
                A_rr_inv[i, j] * ℯ^(im * θ[1] * nodecoordinatedifferences[indxi, indxj, 1])
        end
    elseif dimension == 2
        for i = 1:numbersubassemblednodes, j = 1:numbersubassemblednodes
            indxi = bddc.subassemblednodes[i]
            indxj = bddc.subassemblednodes[j]
            A_rr_inv_nodes[i, j] =
                A_rr_inv[i, j] *
                ℯ^(
                    im * (
                        θ[1] * nodecoordinatedifferences[indxi, indxj, 1] +
                        θ[2] * nodecoordinatedifferences[indxi, indxj, 2]
                    )
                )
        end
    elseif dimension == 3
        for i = 1:numbersubassemblednodes, j = 1:numbersubassemblednodes
            indxi = bddc.subassemblednodes[i]
            indxj = bddc.subassemblednodes[j]
            A_rr_inv_nodes[i, j] =
                A_rr_inv[i, j] *
                ℯ^(
                    im * (
                        θ[1] * nodecoordinatedifferences[indxi, indxj, 1] +
                        θ[2] * nodecoordinatedifferences[indxi, indxj, 2] +
                        θ[3] * nodecoordinatedifferences[indxi, indxj, 3]
                    )
                )
        end
    end

    # mixed subassembled primal matrices
    Â_Πr_nodes = zeros(ComplexF64, numberprimalnodes, numbersubassemblednodes)
    if dimension == 1
        for i = 1:numberprimalnodes, j = 1:numbersubassemblednodes
            indxi = bddc.primalnodes[i]
            indxj = bddc.subassemblednodes[j]
            Â_Πr_nodes[i, j] =
                elementmatrix[indxi, indxj] *
                ℯ^(im * θ[1] * nodecoordinatedifferences[indxi, indxj, 1])
        end
    elseif dimension == 2
        for i = 1:numberprimalnodes, j = 1:numbersubassemblednodes
            indxi = bddc.primalnodes[i]
            indxj = bddc.subassemblednodes[j]
            Â_Πr_nodes[i, j] =
                elementmatrix[indxi, indxj] *
                ℯ^(
                    im * (
                        θ[1] * nodecoordinatedifferences[indxi, indxj, 1] +
                        θ[2] * nodecoordinatedifferences[indxi, indxj, 2]
                    )
                )
        end
    elseif dimension == 3
        for i = 1:numberprimalnodes, j = 1:numbersubassemblednodes
            indxi = bddc.primalnodes[i]
            indxj = bddc.subassemblednodes[j]
            Â_Πr_nodes[i, j] =
                elementmatrix[indxi, indxj] *
                ℯ^(
                    im * (
                        θ[1] * nodecoordinatedifferences[indxi, indxj, 1] +
                        θ[2] * nodecoordinatedifferences[indxi, indxj, 2] +
                        θ[3] * nodecoordinatedifferences[indxi, indxj, 3]
                    )
                )
        end
    end
    Â_Πr_modes = bddc.primalrowmodemap * Â_Πr_nodes

    Â_rΠ_nodes = zeros(ComplexF64, numbersubassemblednodes, numberprimalnodes)
    if dimension == 1
        for i = 1:numbersubassemblednodes, j = 1:numberprimalnodes
            indxi = bddc.subassemblednodes[i]
            indxj = bddc.primalnodes[j]
            Â_rΠ_nodes[i, j] =
                elementmatrix[indxi, indxj] *
                ℯ^(im * θ[1] * nodecoordinatedifferences[indxi, indxj, 1])
        end
    elseif dimension == 2
        for i = 1:numbersubassemblednodes, j = 1:numberprimalnodes
            indxi = bddc.subassemblednodes[i]
            indxj = bddc.primalnodes[j]
            Â_rΠ_nodes[i, j] =
                elementmatrix[indxi, indxj] *
                ℯ^(
                    im * (
                        θ[1] * nodecoordinatedifferences[indxi, indxj, 1] +
                        θ[2] * nodecoordinatedifferences[indxi, indxj, 2]
                    )
                )
        end
    elseif dimension == 3
        for i = 1:numbersubassemblednodes, j = 1:numberprimalnodes
            indxi = bddc.subassemblednodes[i]
            indxj = bddc.primalnodes[j]
            Â_rΠ_nodes[i, j] =
                elementmatrix[indxi, indxj] *
                ℯ^(
                    im * (
                        θ[1] * nodecoordinatedifferences[indxi, indxj, 1] +
                        θ[2] * nodecoordinatedifferences[indxi, indxj, 2] +
                        θ[3] * nodecoordinatedifferences[indxi, indxj, 3]
                    )
                )
        end
    end
    Â_rΠ_modes = Â_rΠ_nodes * bddc.primalcolumnmodemap

    # Schur complement
    Ŝ_Π = bddc.schur
    Ŝ_Π_nodes = zeros(ComplexF64, numberprimalnodes, numberprimalnodes)
    if dimension == 1
        for i = 1:numberprimalnodes, j = 1:numberprimalnodes
            indxi = bddc.primalnodes[i]
            indxj = bddc.primalnodes[j]
            Ŝ_Π_nodes[i, j] =
                Ŝ_Π[i, j] * ℯ^(im * θ[1] * nodecoordinatedifferences[indxi, indxj, 1])
        end
    elseif dimension == 2
        for i = 1:numberprimalnodes, j = 1:numberprimalnodes
            indxi = bddc.primalnodes[i]
            indxj = bddc.primalnodes[j]
            Ŝ_Π_nodes[i, j] =
                Ŝ_Π[i, j] *
                ℯ^(
                    im * (
                        θ[1] * nodecoordinatedifferences[indxi, indxj, 1] +
                        θ[2] * nodecoordinatedifferences[indxi, indxj, 2]
                    )
                )
        end
    elseif dimension == 3
        for i = 1:numberprimalnodes, j = 1:numberprimalnodes
            indxi = bddc.primalnodes[i]
            indxj = bddc.primalnodes[j]
            Ŝ_Π_nodes[i, j] =
                Ŝ_Π[i, j] *
                ℯ^(
                    im * (
                        θ[1] * nodecoordinatedifferences[indxi, indxj, 1] +
                        θ[2] * nodecoordinatedifferences[indxi, indxj, 2] +
                        θ[3] * nodecoordinatedifferences[indxi, indxj, 3]
                    )
                )
        end
    end
    Ŝ_Π_inv_modes = (bddc.primalrowmodemap * Ŝ_Π_nodes * bddc.primalcolumnmodemap)^-1

    # subassembled nodes primal modes
    Ø = zeros((numberprimalmodes, numbersubassemblednodes))
    K_u_inv = [
        I(numbersubassemblednodes) -A_rr_inv_nodes*Â_rΠ_modes
        Ø I(numberprimalmodes)
    ]
    P_inv = [
        A_rr_inv_nodes transpose(Ø)
        Ø Ŝ_Π_inv_modes
    ]
    K_u_T_inv = [
        I(numbersubassemblednodes) transpose(Ø)
        -Â_Πr_modes*A_rr_inv_nodes I(numberprimalmodes)
    ]
    mixedsubassembled = K_u_inv * P_inv * K_u_T_inv

    # injection
    mixedinjected = transpose(bddc.injection) * mixedsubassembled * bddc.injection

    # return
    return I -
           bddc.rowmodemap * mixedinjected * bddc.columnmodemap *
           computesymbols(bddc.fineoperator, θ)
end

# ------------------------------------------------------------------------------
