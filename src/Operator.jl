# ---------------------------------------------------------------------------------------------------------------------
# Finite element operators
# ---------------------------------------------------------------------------------------------------------------------

"""
Finite element operator comprising of a weak form and bases
"""
struct Operator
    weakform::Function
    mesh::Mesh
    inputs::Array{OperatorField}
    outputs::Array{OperatorField}
end

# ---------------------------------------------------------------------------------------------------------------------
# Data for computing symbols
# ---------------------------------------------------------------------------------------------------------------------

stencildict = Dict{Operator,Array{Float64}}()

"""
    getstencil()

Compute or retrieve the stencil of operator for computing the symbol
"""
function getstencil(operator::Operator)
    iscomputed = haskey(stencildict, operator)

    if iscomputed
        # retrieve and return
        return stencildict[operator]
    else
        # compute, store, and return
        return Float64[]
    end
end

# ---------------------------------------------------------------------------------------------------------------------
