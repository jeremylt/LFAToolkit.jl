# ------------------------------------------------------------------------------
# test suite
# ------------------------------------------------------------------------------

using Test, Documenter, LFAToolkit, LinearAlgebra
DocMeta.setdocmeta!(LFAToolkit, :DocTestSetup, :(using LFAToolkit); recursive = true)

# ------------------------------------------------------------------------------
# documentation tests
# ------------------------------------------------------------------------------

@testset "  documentation tests                        " begin
    doctest(LFAToolkit; manual = false)
end

# ------------------------------------------------------------------------------
# mass matrix example
# ------------------------------------------------------------------------------

@testset "  ex001: mass example                        " begin
    include("../examples/ex001_mass.jl")

    @test min(eigenvalues...) ≈ 0.008379422444571976
    @test max(eigenvalues...) ≈ 0.17361111111111088
end

# ------------------------------------------------------------------------------
# diffusion operator example
# ------------------------------------------------------------------------------

@testset "  ex002: diffusion example                   " begin
    include("../examples/ex002_diffusion.jl")

    @test min(eigenvalues...) ≈ 0.3321136902208588
    @test max(eigenvalues...) ≈ 2.131118740197114
end

# ------------------------------------------------------------------------------
# Jacobi smoother example
# ------------------------------------------------------------------------------

@testset "  ex101: Jacobi example                      " begin
    include("../examples/ex101_jacobi.jl")

    @test min(eigenvalues...) ≈ -0.6289239142744161
    @test max(eigenvalues...) ≈ 0.6405931989084651
end

# ------------------------------------------------------------------------------
# Chebyshev smoother example
# ------------------------------------------------------------------------------

@testset "  ex111: Chebyshev example                   " begin
    include("../examples/ex111_chebyshev.jl")

    @test min(eigenvalues...) ≈ -0.272875736279538
    @test max(eigenvalues...) ≈ 0.26048711552603726
end

# ------------------------------------------------------------------------------
# p-multigrid example
# ------------------------------------------------------------------------------

@testset "  ex201: p-multigrid example                 " begin
    include("../examples/ex201_pmultigrid.jl")

    @test max(eigenvalues...) ≈ 0.026444090458920978
end

# ------------------------------------------------------------------------------
# p-multigrid multilevel example
# ------------------------------------------------------------------------------

@testset "  ex202: p-multigrid multilevel example      " begin
    include("../examples/ex202_pmultigrid_multilevel.jl")

    @test max(eigenvalues...) ≈ 0.07661416476317891
end

# ------------------------------------------------------------------------------
# h-multigrid example
# ------------------------------------------------------------------------------

@testset "  ex211: h-multigrid example                 " begin
    include("../examples/ex211_hmultigrid.jl")

    @test max(eigenvalues...) ≈ 0.03791330252587746
end

# ------------------------------------------------------------------------------
# h-multigrid multilevel example
# ------------------------------------------------------------------------------

@testset "  ex212: h-multigrid multilevel example      " begin
    include("../examples/ex212_hmultigrid_multilevel.jl")

    @test max(eigenvalues...) ≈ 0.06922195649613641
end

# ------------------------------------------------------------------------------
# lumped BDDC example
# ------------------------------------------------------------------------------

@testset "  ex221: lumped BDDC example                 " begin
    include("../examples/ex221_lumped_bddc.jl")

    @test min(eigenvalues...) ≈ -0.19999999999999862
    @test max(eigenvalues...) ≈ 0.8000000000000009
end

# ------------------------------------------------------------------------------
# lumped BDDC on macro elements example
# ------------------------------------------------------------------------------

@testset "  ex222: lumped BDDC macro element example   " begin
    include("../examples/ex222_lumped_bddc.jl")

    @test min(eigenvalues...) ≈ 0.11129065897024093
    @test max(eigenvalues...) ≈ 0.8000000000000005
end

# ------------------------------------------------------------------------------
# Dirichlet BDDC example
# ------------------------------------------------------------------------------

@testset "  ex223: Dirichlet BDDC example              " begin
    include("../examples/ex223_dirichlet_bddc.jl")

    @test min(eigenvalues...) ≈ -0.19999999999999862
    @test max(eigenvalues...) ≈ 0.8000000010201185
end

# ------------------------------------------------------------------------------
# Dirichlet BDDC on macro elements example
# ------------------------------------------------------------------------------

@testset "  ex224: Dirichlet BDDC macro element example" begin
    include("../examples/ex224_dirichlet_bddc.jl")

    @test min(eigenvalues...) ≈ 0.1112906589702409
    @test max(eigenvalues...) ≈ 0.8000000003853934
end

# ------------------------------------------------------------------------------
# linear elasticity example
# ------------------------------------------------------------------------------

@testset "  ex301: linear elasticity example           " begin
    include("../examples/ex301_linear_elasticity.jl")

    @test max(eigenvalues...) ≈ 0.07359884745780573
end

# ------------------------------------------------------------------------------
# hyperelasticity example
# ------------------------------------------------------------------------------

@testset "  ex311: hyperelasticity example             " begin
    include("../examples/ex311_hyperelasticity.jl")

    @test max(eigenvalues...) ≈ 0.07818657329883297
end

# ------------------------------------------------------------------------------
