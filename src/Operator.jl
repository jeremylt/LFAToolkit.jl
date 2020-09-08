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

stencildict = Dict{Operator,Array{Float64}}();

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
        # -- collect info on operator
        bases = []
        weakforminputs = []
        numbernodes = 0
        numberquadraturepoints = 0
        weightindex = 0
        quadratureweights = []
        # ---- inputs
        for input in operator.inputs
            # ------ number of nodes
            if !(input.basis in bases)
                push!(bases, input.basis)
                numbernodes += getnumbernodes(input.basis)
            end
            # ------ number of quadrature points
            if numberquadraturepoints == 0
                numberquadraturepoints = getnumberquadraturepoints(input.basis)
            else
                if numberquadraturepoints != getnumberquadraturepoints(input.basis)
                    throw(DomanError(
                        numberquadraturepoints,
                        "All bases much have matching quadrature spaces",
                    )) # COV_EXCL_LINE
                end
            end
            # ------ input mode
            if input.evaluationmode == EvaluationMode.interpolation
                push!(weakforminputs, zeros(input.basis.numbercomponents))
            elseif input.evaluationmode == EvaluationMode.gradient
                push!(
                    weakforminputs,
                    zeros(input.basis.numbercomponents * input.basis.dimension),
                )
            else
                push!(weakforminputs, 0.0)
            end
            if input.evaluationmode == EvaluationMode.quadratureweights
                weightindex = findfirst(isequal(input), operator.inputs)
            end
        end
        if weightindex != 0
            quadratureweights = getquadratureweights(operator.inputs[weightindex].basis)
        end
        # ---- outputs
        for output in operator.outputs
            # ------ number of quadrature points
            if numberquadraturepoints != getnumberquadraturepoints(output.basis)
                throw(DomanError(
                    numberquadraturepoints,
                    "All bases much have matching quadrature spaces",
                )) # COV_EXCL_LINE
            end
        end

        # -- QFunction matrix
        D = spzeros(numberquadraturepoints, numberquadraturepoints)
        for q = 1:numberquadraturepoints
            # ---- set quadrature weight
            if weightindex != 0
                weakforminputs[weightindex] = quadratureweights[q]
            end
            # ---- loop over inputs
            for i = 1:length(operator.inputs)
                input = operator.inputs[i]
                fields = 0
                if input.evaluationmode == EvaluationMode.interpolation
                    fields = input.basis.numbercomponents
                elseif input.evaluationmode == EvaluationMode.gradient
                    fields = input.basis.numbercomponents * input.basis.dimension
                end
                # ------ fill sparse matrix
                for j = 1:fields
                    weakforminputs[i][j] = 1.0
                    outputs = operator.weakform(weakforminputs...)
                    weakforminputs[i][j] = 0.0
                end
            end
        end

        # -- multiply B^T D B

        return Float64[]
    end
end

# ---------------------------------------------------------------------------------------------------------------------
