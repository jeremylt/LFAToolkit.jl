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
end
maxeigenvalues = DataFrame()

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
    numberruns = 100
    maxeigenvalue = 0
    θ_min = -π / 2
    θ_min_high = π / 2
    θ_max = 3π / 2

    # -- compute
    mineigenvalue = 1.0
    ω_minegenvalue = -1
    θ_minegenvalue = -1
    for ω = 0.50:0.01:1.05
        maxeigenvalue = 0
        ω_maxegenvalue = -1
        θ_maxegenvalue = -1
        # -- 1D --
        if dimension == 1
            for i = 1:numberruns
                θ = [θ_min + (θ_max - θ_min) * i / numberruns]
                if abs(θ[1] % 2π) > π / 128 && θ[1] > θ_min_high
                    A = computesymbols(jacobi, [ω], θ)
                    eigenvalues = [abs(val) for val in eigvals(A)]
                    currentmaxeigenvalue = max(eigenvalues...)
                    if (currentmaxeigenvalue > maxeigenvalue)
                        maxeigenvalue = currentmaxeigenvalue
                        ω_maxegenvalue = ω
                        θ_maxegenvalue = θ
                    end
                end
            end
            # -- 2D --
        elseif dimension == 2
            for i = 1:numberruns, j = 1:numberruns
                θ = [
                    θ_min + (θ_max - θ_min) * i / numberruns,
                    θ_min + (θ_max - θ_min) * j / numberruns,
                ]
                if sqrt(abs(θ[1] % 2π)^2 + abs(θ[2] % 2π)^2) > π / 128 &&
                   (θ[1] > θ_min_high || θ[2] > θ_min_high)
                    A = computesymbols(jacobi, [ω], θ)
                    eigenvalues = [abs(val) for val in eigvals(A)]
                    currentmaxeigenvalue = max(eigenvalues...)
                    if (currentmaxeigenvalue > maxeigenvalue)
                        maxeigenvalue = currentmaxeigenvalue
                        ω_maxegenvalue = ω
                        θ_maxegenvalue = θ
                    end
                end
            end
        end
        if (maxeigenvalue < mineigenvalue)
            mineigenvalue = maxeigenvalue
            ω_minegenvalue = ω_maxegenvalue
            θ_minegenvalue = θ_maxegenvalue / π
        end
    end
    append!(
        maxeigenvalues,
        DataFrame(p = p, ω = ω_minegenvalue, θ = θ_minegenvalue, rho = mineigenvalue),
    )
end

CSV.write("jacobi-smoothing-factor.csv", maxeigenvalues)
