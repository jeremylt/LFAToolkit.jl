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
smoothingfactors = DataFrame()

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
    ω_range = 0.50:0.01:1.05
    num_ω = max(size(ω_range)...)

    # -- compute
    maxeigenvalue = zeros(num_ω)
    θ_maxeigenvalue = -1 * ones(num_ω, dimension)
    # -- 1D --
    if dimension == 1
        for i = 1:numbersteps
            θ = [θ_range[i]]
            if abs(θ[1]) > π / 128 && θ[1] > θ_min_high
                A = computesymbols(jacobi, [1.0], θ)
                for w = 1:num_ω
                    eigenvalues = [abs(val) for val in eigvals(I - ω_range[w] * (I - A))]
                    currentmaxeigenvalue = max(eigenvalues...)
                    if (currentmaxeigenvalue > maxeigenvalue[w])
                        maxeigenvalue[w] = currentmaxeigenvalue
                        θ_maxeigenvalue[w, :] = θ / π
                    end
                end
            end
        end
        # -- 2D --
    elseif dimension == 2
        for i = 1:numbersteps, j = 1:numbersteps
            θ = [θ_range[i], θ_range[j]]
            if sqrt(abs(θ[1])^2 + abs(θ[2])^2) > π / 128 &&
               (θ[1] > θ_min_high || θ[2] > θ_min_high)
                A = computesymbols(jacobi, [1.0], θ)
                for w = 1:num_ω
                    eigenvalues = [abs(val) for val in eigvals(I - ω_range[w] * (I - A))]
                    currentmaxeigenvalue = max(eigenvalues...)
                    if (currentmaxeigenvalue > maxeigenvalue[w])
                        maxeigenvalue[w] = currentmaxeigenvalue
                        θ_maxeigenvalue[w, :] = θ / π
                    end
                end
            end
        end
        # -- 3D --
    elseif dimension == 3
        for i = 1:numbersteps, j = 1:numbersteps, k = 1:numbersteps
            θ = [θ_range[i], θ_range[j], θ_range[k]]
            if sqrt(abs(θ[1])^2 + abs(θ[2])^2 + abs(θ[3])^2) > π / 128 &&
               (θ[1] > θ_min_high || θ[2] > θ_min_high || θ[3] > θ_min_high)
                A = computesymbols(jacobi, [1.0], θ)
                for w = 1:num_ω
                    eigenvalues = [abs(val) for val in eigvals(I - ω_range[w] * (I - A))]
                    currentmaxeigenvalue = max(eigenvalues...)
                    if (currentmaxeigenvalue > maxeigenvalue[w])
                        maxeigenvalue[w] = currentmaxeigenvalue
                        θ_maxeigenvalue[w, :] = θ / π
                    end
                end
            end
        end
    end
    (_, ω_index) = findmin(maxeigenvalue)
    append!(
        smoothingfactors,
        DataFrame(
            p = p,
            ω = ω_range[ω_index],
            θ = θ_maxeigenvalue[ω_index, :],
            ρ = maxeigenvalue[ω_index],
        ),
    )
end

CSV.write("jacobi-smoothing-factor.csv", smoothingfactors)
