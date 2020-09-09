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

```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
basis = TensorH1LagrangeBasis(4, 4, 2, 1);
    
function massweakform(u::Array{Float64,1}, w::Array{Float64})
    v = u * w[1]
    return [v]
end
    
# mass operator
inputs = [
    OperatorField(basis, EvaluationMode.interpolation),
    OperatorField(basis, EvaluationMode.quadratureweights),
];
outputs = [OperatorField(basis, EvaluationMode.interpolation)];
mass = Operator(massweakform, mesh, inputs, outputs);
    
# stencil computation
stencil = getstencil(mass);

# verify
u = ones(4*4);
v = stencil * u;
    
total = sum(v);
if abs(total - 4.0) > 1e-14
    println("Incorrect mass matrix");
end

# test caching
stencil = getstencil(mass)
v = stencil * u;
    
total = sum(v);
if abs(total - 4.0) > 1e-14
    println("Incorrect mass matrix");
end
    
# output
    
```

```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
basis = TensorH1LagrangeBasis(4, 4, 2, 1);
    
function diffusionweakform(du::Array{Float64,1}, w::Array{Float64})
    dv = du * w[1]
    return [dv]
end
    
# diffusion operator
inputs = [
    OperatorField(basis, EvaluationMode.gradient),
    OperatorField(basis, EvaluationMode.quadratureweights),
];
outputs = [OperatorField(basis, EvaluationMode.gradient)];
diffusion = Operator(diffusionweakform, mesh, inputs, outputs);
    
# stencil computation
stencil = getstencil(diffusion);
    
# verify
u = ones(4*4);
v = stencil * u;
    
total = sum(v);
if abs(total) > 1e-14
    println("Incorrect diffusion matrix");
end
    
# output
    
```
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
        numberquadratureinputs = 0
        weightinputindex = 0
        quadratureweights = []
        B = []
        Bt = []
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
                    # COV_EXCL_START
                    throw(DomanError(
                        numberquadraturepoints,
                        "All bases much have matching quadrature spaces",
                    ))
                    # COV_EXCL_STOP
                end
            end
            # ------ input mode
            if input.evaluationmode == EvaluationMode.interpolation
                push!(weakforminputs, zeros(input.basis.numbercomponents))
                numberquadratureinputs += input.basis.numbercomponents
                B = B == [] ? getinterpolation(input.basis) :
                    [B; getinterpolation(input.basis)]
            elseif input.evaluationmode == EvaluationMode.gradient
                push!(
                    weakforminputs,
                    zeros(input.basis.numbercomponents * input.basis.dimension),
                )
                numberquadratureinputs +=
                    input.basis.numbercomponents * input.basis.dimension
                B = B == [] ? getgradient(input.basis) : [B; getgradient(input.basis)]
            elseif input.evaluationmode == EvaluationMode.quadratureweights
                push!(weakforminputs, zeros(1))
                weightinputindex = findfirst(isequal(input), operator.inputs)
            end
        end
        if weightinputindex != 0
            quadratureweights =
                getquadratureweights(operator.inputs[weightinputindex].basis)
        end
        # ---- outputs
        for output in operator.outputs
            # ------ number of quadrature points
            if numberquadraturepoints != getnumberquadraturepoints(output.basis)
                # COV_EXCL_START
                throw(DomanError(
                    numberquadraturepoints,
                    "All bases much have matching quadrature spaces",
                ))
                # COV_EXCL_STOP
            end
            # ------ output mode
            if output.evaluationmode == EvaluationMode.interpolation
                Bt = Bt == [] ? getinterpolation(output.basis) :
                    [Bt; getinterpolation(output.basis)]
            elseif output.evaluationmode == EvaluationMode.gradient
                Bt = Bt == [] ? getgradient(output.basis) : [Bt; getgradient(output.basis)]
                # COV_EXCL_START
            elseif output.evaluationmode == EvaluationMode.quadratureweights
                throw(ArgumentError("quadratureweights is not a valid ouput field evaluation mode"))
                # COV_EXCL_STOP
            end
        end
        Bt = transpose(Bt)

        # -- QFunction matrix
        D = spzeros(
            numberquadratureinputs * numberquadraturepoints,
            numberquadratureinputs * numberquadraturepoints,
        )
        for q = 1:numberquadraturepoints
            # ---- set quadrature weight
            if weightinputindex != 0
                weakforminputs[weightinputindex][1] = quadratureweights[q]
            end
            # ---- loop over inputs
            currentfieldin = 0
            for i = 1:length(operator.inputs)
                input = operator.inputs[i]
                numberfieldsin = 0
                if input.evaluationmode == EvaluationMode.interpolation
                    numberfieldsin = input.basis.numbercomponents
                elseif input.evaluationmode == EvaluationMode.gradient
                    numberfieldsin = input.basis.numbercomponents * input.basis.dimension
                end
                # ------ fill sparse matrix
                for j = 1:numberfieldsin
                    # -------- run user weak form function
                    weakforminputs[i][j] = 1.0
                    outputs = operator.weakform(weakforminputs...)
                    weakforminputs[i][j] = 0.0

                    # -------- store outputs
                    currentfieldout = 0
                    for k = 1:length(operator.outputs)
                        output = operator.outputs[i]
                        numberfieldsout = 0
                        if output.evaluationmode == EvaluationMode.interpolation
                            numberfieldsout = output.basis.numbercomponents
                        elseif output.evaluationmode == EvaluationMode.gradient
                            numberfieldsout =
                                output.basis.numbercomponents * output.basis.dimension
                        end
                        for l = 1:numberfieldsout
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

        # -- multiply A = B^T D B and store
        stencil = Bt * D * B
        stencildict[operator] = stencil

        # -- return
        return stencil
    end
end

# ---------------------------------------------------------------------------------------------------------------------
