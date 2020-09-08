using Test, Documenter, LFAToolkit
DocMeta.setdocmeta!(LFAToolkit, :DocTestSetup, :(using LFAToolkit); recursive = true)

@testset "LFAToolkit" begin

    # ---------------------------------------------------------------------------------------------------------------------
    # Documentation
    # ---------------------------------------------------------------------------------------------------------------------
    doctest(LFAToolkit; manual = false)

    # ---------------------------------------------------------------------------------------------------------------------
    # Full example
    # ---------------------------------------------------------------------------------------------------------------------

    mesh = Mesh2D(1.0, 1.0)
    basis = TensorH1LagrangeBasis(4, 4, 2, 1)

    # Mass operator
    inputs = [
        OperatorField(basis, EvaluationMode.interpolation),
        OperatorField(basis, EvaluationMode.quadratureweights),
    ]
    outputs = [OperatorField(basis, EvaluationMode.interpolation)]

    function massweakform(u::Float64, w::Float64)
        v = u * w
        return [v]
    end

    mass = Operator(massweakform, mesh, inputs, outputs)

    stencil = getstencil(mass)

    # Diffusion operator
    inputs = [
        OperatorField(basis, EvaluationMode.gradient),
        OperatorField(basis, EvaluationMode.quadratureweights),
    ]
    outputs = [OperatorField(basis, EvaluationMode.gradient)]

    function diffusionweakform(du::Array{Float64,1}, w::Float64)
        dv = du * w
        return [dv]
    end

    diffusion = Operator(diffusionweakform, mesh, inputs, outputs)

    stencil = getstencil(diffusion)

end # testset

# ---------------------------------------------------------------------------------------------------------------------
