using Test, Documenter, LFAToolkit
DocMeta.setdocmeta!(LFAToolkit, :DocTestSetup, :(using LFAToolkit); recursive = true)

@testset "LFAToolkit" begin

    # ---------------------------------------------------------------------------------------------------------------------
    # documentation tests
    # ---------------------------------------------------------------------------------------------------------------------
    doctest(LFAToolkit; manual = false)

    # ---------------------------------------------------------------------------------------------------------------------
    # mass matrix example
    # ---------------------------------------------------------------------------------------------------------------------

    # setup
    mesh = Mesh2D(1.0, 1.0)
    basis = TensorH1LagrangeBasis(4, 4, 2)

    function massweakform(u::Array{Float64}, w::Array{Float64})
        v = u * w[1]
        return [v]
    end

    # mass operator
    inputs = [
        OperatorField(basis, [EvaluationMode.interpolation]),
        OperatorField(basis, [EvaluationMode.quadratureweights]),
    ]
    outputs = [OperatorField(basis, [EvaluationMode.interpolation])]
    mass = Operator(massweakform, mesh, inputs, outputs)

    # compute operator symbols
    A = computesymbols(mass, π, π)

    # ---------------------------------------------------------------------------------------------------------------------
    # diffusion operator example
    # ---------------------------------------------------------------------------------------------------------------------

    # diffusion setup
    function diffusionweakform(du::Array{Float64}, w::Array{Float64})
        dv = du * w[1]
        return [dv]
    end

    # diffusion operator
    inputs = [
        OperatorField(basis, [EvaluationMode.gradient]),
        OperatorField(basis, [EvaluationMode.quadratureweights]),
    ]
    outputs = [OperatorField(basis, [EvaluationMode.gradient])]
    diffusion = Operator(diffusionweakform, mesh, inputs, outputs)

    # compute operator symbols
    A = computesymbols(diffusion, π, π)

end # testset

# ---------------------------------------------------------------------------------------------------------------------
