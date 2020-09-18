# ---------------------------------------------------------------------------------------------------------------------
# Finite element operators
# ---------------------------------------------------------------------------------------------------------------------

"""
```julia
Operator(
    weakform,
    mesh,
    inputs,
    outputs
)
```

Finite element operator comprising of a weak form and bases

# Arguments:
- `weakform`: user provided function that represents weak form at quadrature points
- `mesh`:     mesh object with deformation in each dimension
- `inputs`:   array of operator input fields
- `outputs`:  array of operator output fields

# Returns:
- Finite element operator object
"""
mutable struct Operator
    # never changed
    weakform::Function
    mesh::Mesh
    inputs::Array{OperatorField}
    outputs::Array{OperatorField}

    # empty until assembled
    stencil::Array{Float64,2}

    # constructor
    Operator(weakform, mesh, inputs, outputs) = (
        # check inputs vaidity
        for input in inputs
            if length(input.evaluationmodes) > 1 &&
               EvaluationMode.quadratureweights in input.evaluationmodes
                error("quadrature weights must be a separate input") # COV_EXCL_LINE
            end
        end;

        # check outputs valididy
        for output in outputs
            if EvaluationMode.quadratureweights in output.evaluationmodes
                error("quadrature weights is not a valid output") # COV_EXCL_LINE
            end
        end;

        # constructor
        new(weakform, mesh, inputs, outputs)
    )
end

# ---------------------------------------------------------------------------------------------------------------------
# Data for computing symbols
# ---------------------------------------------------------------------------------------------------------------------

"""
```julia
getstencil(operator)
```

Compute or retrieve the stencil of operator for computing the symbol

# Arguments:
- `operator`: operator to compute element stencil

# Returns:
- Assembled element matrix

# Mass matrix example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
basis = TensorH1LagrangeBasis(4, 4, 2);
    
function massweakform(u::Array{Float64}, w::Array{Float64})
    v = u * w[1]
    return [v]
end
    
# mass operator
inputs = [
    OperatorField(basis, [EvaluationMode.interpolation]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.interpolation])];
mass = Operator(massweakform, mesh, inputs, outputs);
    
# stencil computation
stencil = LFAToolkit.getstencil(mass);
stencil = mass.stencil; # either syntax works

# verify
u = ones(4*4);
v = stencil * u;
    
total = sum(v);
@assert abs(total - 4.0) < 1e-14
    
# output

```

# Diffusion matrix example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
basis = TensorH1LagrangeBasis(4, 4, 2);
    
function diffusionweakform(du::Array{Float64}, w::Array{Float64})
    dv = du * w[1]
    return [dv]
end
    
# diffusion operator
inputs = [
    OperatorField(basis, [EvaluationMode.gradient]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.gradient])];
diffusion = Operator(diffusionweakform, mesh, inputs, outputs);
    
# stencil computation
stencil = LFAToolkit.getstencil(diffusion);
stencil = diffusion.stencil; # either syntax works
    
# verify
u = ones(4*4);
v = stencil * u;
    
total = sum(v);
@assert abs(total) < 1e-14
    
# output

```
"""
function getstencil(operator::Operator)
    # assemble if needed
    if !isdefined(operator, :stencil)
        # -- collect info on operator
        weakforminputs = []
        numbernodes = 0
        numberquadraturepoints = 0
        numberquadratureinputs = 0
        weightinputindex = 0
        quadratureweights = []
        Bblocks = []
        Btblocks = []
        # ---- inputs
        for input in operator.inputs
            # ------ number of nodes
            if input.evaluationmodes[1] != EvaluationMode.quadratureweights
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
            # ------ input evaluation modes
            if input.evaluationmodes[1] == EvaluationMode.quadratureweights
                push!(weakforminputs, zeros(1))
                weightinputindex = findfirst(isequal(input), operator.inputs)
            else
                numbermodes = 0
                Bcurrent = []
                for mode in input.evaluationmodes
                    if mode == EvaluationMode.interpolation
                        numbermodes += 1
                        numberquadratureinputs += 1
                        Bcurrent =
                            Bcurrent == [] ? getinterpolation(input.basis) :
                            [Bcurrent; getinterpolation(input.basis)]
                    elseif mode == EvaluationMode.gradient
                        numbermodes += input.basis.dimension
                        numberquadratureinputs += input.basis.dimension
                        Bcurrent =
                            Bcurrent == [] ? getgradient(input.basis) :
                            [Bcurrent; getgradient(input.basis)]
                    end
                end
                push!(Bblocks, Bcurrent)
                push!(weakforminputs, zeros(numbermodes))
            end
        end
        # ------ input basis matrix
        B = spzeros(numberquadratureinputs * numberquadraturepoints, numbernodes)
        currentrow = 1
        currentcolumn = 1
        for Bblock in Bblocks
            B[currentrow:size(Bblock)[1], currentcolumn:size(Bblock)[2]] = Bblock
            currentrow += size(Bblock)[1]
            currentcolumn += size(Bblock)[2]
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
            # ------ output evaluation modes
            numbermodes = 0
            Btcurrent = []
            for mode in output.evaluationmodes
                if mode == EvaluationMode.interpolation
                    Btcurrent =
                        Btcurrent == [] ? getinterpolation(output.basis) :
                        [Btcurrent; getinterpolation(output.basis)]
                elseif mode == EvaluationMode.gradient
                    Btcurrent =
                        Btcurrent == [] ? getgradient(output.basis) :
                        [Btcurrent; getgradient(output.basis)]
                    # Note, quadrature weights checked in constructor
                end
            end
            push!(Btblocks, Btcurrent)
        end
        # ------ output basis matrix
        Bt = spzeros(numberquadratureinputs * numberquadraturepoints, numbernodes)
        currentrow = 1
        currentcolumn = 1
        for Btblock in Btblocks
            Bt[currentrow:size(Btblock)[1], currentcolumn:size(Btblock)[2]] = Btblock
            currentrow += size(Btblock)[1]
            currentcolumn += size(Btblock)[2]
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
                if input.evaluationmodes[1] == EvaluationMode.quadratureweights
                    break
                end
                # ------ fill sparse matrix
                for j = 1:length(input.evaluationmodes)
                    # -------- run user weak form function
                    weakforminputs[i][j] = 1.0
                    outputs = operator.weakform(weakforminputs...)
                    weakforminputs[i][j] = 0.0

                    # -------- store outputs
                    currentfieldout = 0
                    for k = 1:length(operator.outputs)
                        output = operator.outputs[k]
                        for l = 1:length(output.evaluationmodes)
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
        operator.stencil = stencil
    end

    # return
    return getfield(operator, :stencil)
end

# ---------------------------------------------------------------------------------------------------------------------
# get/set property
# ---------------------------------------------------------------------------------------------------------------------

function Base.getproperty(operator::Operator, f::Symbol)
    if f == :stencil
        return getstencil(operator)
    else
        return getfield(operator, f)
    end
end

# ---------------------------------------------------------------------------------------------------------------------
