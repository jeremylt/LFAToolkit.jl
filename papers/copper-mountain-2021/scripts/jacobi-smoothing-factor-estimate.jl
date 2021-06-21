# dependencies
using LFAToolkit
using LinearAlgebra
using CSV
using DataFrames

# setup
dimension = 1
numbercomponents = 1
mesh = []
if dimension == 1
    mesh = Mesh1D(1.0)
elseif dimension == 2
    mesh = Mesh2D(1.0, 1.0)
elseif dimension == 3
    mesh = Mesh3D(1.0, 1.0, 1.0)
end
smoothingparameters = DataFrame()

# test range
for P = 1:4
    println("p = ", 2^P)
    # setup
    # -- bases
    p = 2^P

    # -- diffusion operator
    diffusion = GalleryOperator("diffusion", p + 1, p + 1, mesh)

    # -- Jacobi smoother
    jacobi = Jacobi(diffusion)

    # compute smoothing factor
    # -- setup
    numbersteps = 100
    maxeigenvalue = 0
    θ_min = -π / 2
    θ_min_high = π / 2
    θ_max = 3π / 2
    θ_step = 2π / numbersteps
    θ_range = θ_min:θ_step:(θ_max-θ_step)
    ω = 1.0

    # -- compute
    λ_minhigh = 100
    λ_maxhigh = -100
    λ_max = -100
    # -- 1D --
    if dimension == 1
        for i = 1:numbersteps
            θ = [θ_range[i]]
            if abs(θ[1]) > π / 128
                A = computesymbols(jacobi, [ω], θ)
                eigenvalues = [abs(val) for val in eigvals(I - A)]
                if (θ[1] > θ_min_high)
                    λ_minhigh = min(λ_minhigh, eigenvalues...)
                    λ_maxhigh = max(λ_maxhigh, eigenvalues...)
                end
                λ_max = max(λ_max, eigenvalues...)
            end
        end
        # -- 2D --
    elseif dimension == 2
        for i = 1:numbersteps, j = 1:numbersteps
            θ = [θ_range[i], θ_range[j]]
            if sqrt(abs(θ[1] % 2π)^2 + abs(θ[2] % 2π)^2) > π / 128
                A = computesymbols(jacobi, [ω], θ)
                eigenvalues = [abs(val) for val in eigvals(I - A)]
                if (θ[1] > θ_min_high || θ[2] > θ_min_high)
                    λ_minhigh = min(λ_minhigh, eigenvalues...)
                    λ_maxhigh = max(λ_maxhigh, eigenvalues...)
                end
                λ_max = max(λ_max, eigenvalues...)
            end
        end
        # -- 3D --
    elseif dimension == 3
        for i = 1:numbersteps, j = 1:numbersteps, k = 1:numbersteps
            θ = [θ_range[i], θ_range[j], θ_range[k]]
            if sqrt(abs(θ[1])^2 + abs(θ[2])^2 + abs(θ[3])^2) > π / 128
                A = computesymbols(jacobi, [ω], θ)
                eigenvalues = [abs(val) for val in eigvals(I - A)]
                if (θ[1] > θ_min_high || θ[2] > θ_min_high)
                    λ_minhigh = min(λ_minhigh, eigenvalues...)
                    λ_maxhigh = max(λ_maxhigh, eigenvalues...)
                end
                λ_max = max(λ_max, eigenvalues...)
            end
        end
    end
    ω_classical = 2 / (λ_minhigh + λ_maxhigh)
    ω_highorder = 2 / (λ_minhigh + λ_max)
    append!(
        smoothingparameters,
        DataFrame(p = p, ω_classical = ω_classical, ω_highorder = ω_highorder),
    )
end

CSV.write("jacobi-smoothing-factor-estimate.csv", smoothingparameters)
