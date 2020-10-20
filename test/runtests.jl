# ------------------------------------------------------------------------------
# test suite
# ------------------------------------------------------------------------------

using Test, Documenter, LFAToolkit, LinearAlgebra
DocMeta.setdocmeta!(LFAToolkit, :DocTestSetup, :(using LFAToolkit); recursive = true)

# ------------------------------------------------------------------------------
# documentation tests
# ------------------------------------------------------------------------------

@testset "documentation tests      " begin
    doctest(LFAToolkit; manual = false)
end

# ------------------------------------------------------------------------------
# mass matrix example
# ------------------------------------------------------------------------------

@testset "ex1 - mass example       " begin
    include("../examples/ex1_mass.jl")

    @test min(eigenvalues...) ≈ 0.008379422444571976
    @test max(eigenvalues...) ≈ 0.17361111111111088
end

# ------------------------------------------------------------------------------
# diffusion operator example
# ------------------------------------------------------------------------------

@testset "ex2 - diffusion example  " begin
    include("../examples/ex2_diffusion.jl")

    @test min(eigenvalues...) ≈ 0.24284729156204987
    @test max(eigenvalues...) ≈ 7.852944737646179
end

# ------------------------------------------------------------------------------
# Jacobi smoother example
# ------------------------------------------------------------------------------

@testset "ex3 - Jacobi example     " begin
    include("../examples/ex3_jacobi.jl")

    @test min(eigenvalues...) ≈ -1.5989685969312784
    @test max(eigenvalues...) ≈ 0.8446129151683509
end

# ------------------------------------------------------------------------------
# p-multigrid example
# ------------------------------------------------------------------------------

@testset "ex4 - p-multigrid example" begin
    include("../examples/ex4_pmultigrid.jl")

    @test min(eigenvalues...) ≈ -15.673827693874575
    @test max(eigenvalues...) ≈ 2.5567005739723823
end

# ------------------------------------------------------------------------------
