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
convergencefactors = DataFrame()

# test range
for fineP = 1:4
    println("fine_p = ", 2^fineP)
    for coarseP = 0:fineP-1
        println("  coarse_p = ", 2^coarseP)
        # setup
        # -- bases
        coarsep = 2^coarseP
        finep = 2^fineP

        ctofbasis = TensorH1LagrangePProlongationBasis(
            coarsep + 1,
            finep + 1,
            numbercomponents,
            dimension,
        )

        # -- diffusion operators
        finediffusion = GalleryOperator("diffusion", finep + 1, finep + 1, mesh)
        coarsediffusion = GalleryOperator("diffusion", coarsep + 1, finep + 1, mesh)

        # -- smoothers
        identity = IdentityPC(finediffusion)
        jacobi = Jacobi(finediffusion)

        # -- p-multigrid preconditioner
        multigrid = PMultigrid(finediffusion, coarsediffusion, identity, [ctofbasis])

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
        v_range = 1:3
        num_v = 3

        # -- compute
        maxeigenvalue = zeros(num_ω, num_v)
        θ_maxeigenvalue = -1 * ones(num_ω, num_v, dimension)
        # -- 1D --
        if dimension == 1
            for i = 1:numbersteps
                θ = [θ_range[i]]
                if abs(θ[1]) > π / 128
                    M = computesymbols(multigrid, [0], [0, 0], θ)
                    A = I - computesymbols(jacobi, [1.0], θ)
                    for w = 1:num_ω, v in v_range
                        S = (I - ω_range[w] * A)^v
                        eigenvalues = [abs(val) for val in eigvals(S * M * S)]
                        currentmaxeigenvalue = max(eigenvalues...)
                        if (currentmaxeigenvalue > maxeigenvalue[w, v])
                            maxeigenvalue[w, v] = currentmaxeigenvalue
                            θ_maxeigenvalue[w, v, :] = θ / π
                        end
                    end
                end
            end
            # -- 2D --
        elseif dimension == 2
            for i = 1:numbersteps, j = 1:numbersteps
                θ = [θ_range[i], θ_range[j]]
                if sqrt(abs(θ[1])^2 + abs(θ[2])^2) > π / 128
                    M = computesymbols(multigrid, [0], [0, 0], θ)
                    A = I - computesymbols(jacobi, [1.0], θ)
                    for w = 1:num_ω, v in v_range
                        S = (I - ω_range[w] * A)^v
                        eigenvalues = [abs(val) for val in eigvals(S * M * S)]
                        currentmaxeigenvalue = max(eigenvalues...)
                        if (currentmaxeigenvalue > maxeigenvalue[w, v])
                            maxeigenvalue[w, v] = currentmaxeigenvalue
                            θ_maxeigenvalue[w, v, :] = θ / π
                        end
                    end
                end
            end
            # -- 3D --
        elseif dimension == 3
            for i = 1:numbersteps, j = 1:numbersteps, k = 1:numbersteps
                θ = [θ_range[i], θ_range[j], θ_range[k]]
                if sqrt(abs(θ[1])^2 + abs(θ[2])^2 + abs(θ[3])^2) > π / 128
                    M = computesymbols(multigrid, [0], [0, 0], θ)
                    A = I - computesymbols(jacobi, [1.0], θ)
                    for w = 1:num_ω, v in v_range
                        S = (I - ω_range[w] * A)^v
                        eigenvalues = [abs(val) for val in eigvals(S * M * S)]
                        currentmaxeigenvalue = max(eigenvalues...)
                        if (currentmaxeigenvalue > maxeigenvalue[w, v])
                            maxeigenvalue[w, v] = currentmaxeigenvalue
                            θ_maxeigenvalue[w, v, :] = θ / π
                        end
                    end
                end
            end
        end
        for v in v_range
            (_, ω_index) = findmin(maxeigenvalue[:, v])
            append!(
                convergencefactors,
                DataFrame(
                    finep = finep,
                    coarsep = coarsep,
                    ω = ω_range[ω_index],
                    θ = θ_maxeigenvalue[ω_index, v, :],
                    v = v,
                    ρ = maxeigenvalue[ω_index, v],
                ),
            )
        end
    end
end

CSV.write("jacobi-smoothing-factor.csv", convergencefactors)
