# ------------------------------------------------------------------------------
# test suite
# ------------------------------------------------------------------------------
using Test, Documenter, LFAToolkit
DocMeta.setdocmeta!(LFAToolkit, :DocTestSetup, :(using LFAToolkit); recursive = true)

# ------------------------------------------------------------------------------
@testset "LFAToolkit" begin

    # --------------------------------------------------------------------------
    # documentation tests
    # --------------------------------------------------------------------------
    doctest(LFAToolkit; manual = false)

    using LinearAlgebra

    # --------------------------------------------------------------------------
    # mass matrix example
    # --------------------------------------------------------------------------

    # setup
    mesh = Mesh2D(1.0, 1.0)
    basis = TensorH1LagrangeBasis(4, 4, 2)

    function massweakform(u::Array{Float64}, w::Array{Float64})
        v = u*w[1]
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

    # verify
    eigenvalues = real(eigvals(A))
    @test min(eigenvalues...) ≈ 0.008379422444571976
    @test max(eigenvalues...) ≈ 0.17361111111111088

    # --------------------------------------------------------------------------
    # diffusion operator example
    # --------------------------------------------------------------------------

    # diffusion setup
    function diffusionweakform(du::Array{Float64}, w::Array{Float64})
        dv = du*w[1]
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

    # verify
    eigenvalues = real(eigvals(A))
    @test min(eigenvalues...) ≈ 0.06071182289051259
    @test max(eigenvalues...) ≈ 1.9632361844115431

    # --------------------------------------------------------------------------
    # jacobi smoother example
    # --------------------------------------------------------------------------

    # jacobi smoother
    jacobi = Jacobi(diffusion)

    # compute operator symbols
    A = computesymbols(jacobi, 1.0, π, π)
    
    # verify
    eigenvalues = real(eigvals(A))
    @test min(eigenvalues...) ≈ -1.5989685969312784
    @test max(eigenvalues...) ≈ 0.8446129151683509

end # testset

# ------------------------------------------------------------------------------
