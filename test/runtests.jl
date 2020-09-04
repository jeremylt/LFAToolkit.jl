using Test, Documenter, LFAToolkit
DocMeta.setdocmeta!(LFAToolkit, :DocTestSetup, :(using LFAToolkit); recursive = true)

@testset "LFAToolkit" begin

    # ---------------------------------------------------------------------------------------------------------------------
    # Documentation
    # ---------------------------------------------------------------------------------------------------------------------
    doctest(LFAToolkit; manual = false)

    # ---------------------------------------------------------------------------------------------------------------------
    # Basis construction
    # ---------------------------------------------------------------------------------------------------------------------

    p = 4
    q = 4
    dimension = 1
    numbercomponents = 1
    nodes = [-1.0, -sqrt(5.0) / 5.0, sqrt(5.0) / 5.0, 1.0]
    quadraturepoints = [-0.86113631, -0.33998104, 0.33998104, 0.86113631]
    quadratureweights = [0.34785485, 0.65214515, 0.65214515, 0.34785485]
    interpolation = [
        0.62994317 0.47255875 -0.14950343 0.04700152
        -0.07069480 0.97297619 0.13253993 -0.03482132
        -0.03482132 0.13253993 0.97297619 -0.07069480
        0.04700152 -0.14950343 0.47255875 0.62994317
    ]
    gradient = [
        -2.34183742 2.78794489 -0.63510411 0.18899664
        -0.51670214 -0.48795249 1.33790510 -0.33325047
        0.33325047 -1.33790510 0.48795249 0.51670214
        -0.18899664 0.63510411 -2.78794489 2.34183742
    ]

    basis = TensorH1LagrangeBasis(p, q, dimension, numbercomponents)

    @test basis.p1d == p
    @test basis.q1d == q
    @test basis.dimension == dimension
    @test basis.numbercomponents == numbercomponents

    tol = 1e-7
    for i = 1:p
        @test abs(basis.nodes1d[i] - nodes[i]) < tol
    end
    for i = 1:q
        @test abs(basis.quadraturepoints1d[i] - quadraturepoints[i]) < tol
    end
    for i = 1:q, j = 1:p
        @test abs(basis.interpolation1d[i, j] - interpolation[i, j]) < tol
        @test abs(basis.gradient1d[i, j] - gradient[i, j]) < tol
    end

    # ---------------------------------------------------------------------------------------------------------------------
    # Operator construction
    # ---------------------------------------------------------------------------------------------------------------------

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

    mass = Operator(massweakform, inputs, outputs)

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

    diffusion = Operator(diffusionweakform, inputs, outputs)

    stencil = getstencil(diffusion)

end # testset

# ---------------------------------------------------------------------------------------------------------------------
