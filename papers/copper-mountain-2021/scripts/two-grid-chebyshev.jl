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
        chebyshev = Chebyshev(finediffusion)

        # -- p-multigrid preconditioner
        multigrid = PMultigrid(finediffusion, coarsediffusion, identity, [ctofbasis])

        # compute smoothing factor
        # -- setup
        numbersteps = 8
        θ_min = -π / 2
        θ_max = 3π / 2
        θ_step = 2π / numbersteps
        θ_range = θ_min:θ_step:(θ_max-θ_step)
        if dimension == 3
            numbersteps = 8
            θ_step = 2π / numbersteps
            θ_range = θ_min:θ_step:(θ_max-θ_step)
        end

        # -- compute
        maxeigenvalue = zeros(4)
        θ_maxeigenvalue = -1 * ones(4, dimension)
        # -- 1D --
        if dimension == 1
            for i = 1:numbersteps
                θ = [θ_range[i]]
                if abs(θ[1]) > π / 128
                    M = computesymbols(multigrid, [0], [0, 0], θ)
                    for k = 1:4
                        S = computesymbols(chebyshev, [k], θ)
                        eigenvalues = [abs(val) for val in eigvals(S * M * S)]
                        currentmaxeigenvalue = max(eigenvalues...)
                        if (currentmaxeigenvalue > maxeigenvalue[k])
                            maxeigenvalue[k] = currentmaxeigenvalue
                            θ_maxeigenvalue[k, :] = θ / π
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
                    for k = 1:4
                        S = computesymbols(chebyshev, [k], θ)
                        eigenvalues = [abs(val) for val in eigvals(S * M * S)]
                        currentmaxeigenvalue = max(eigenvalues...)
                        if (currentmaxeigenvalue > maxeigenvalue[k])
                            maxeigenvalue[k] = currentmaxeigenvalue
                            θ_maxeigenvalue[k, :] = θ / π
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
                    for k = 1:4
                        eigenvalueestimates =
                            [0.0, 1.3393, 1.9893, 2.1993, 2.2932, 2.3463, 2.3823, 2.4102]
                        eigenvalue = eigenvalueestimates[finep]
                        S = []
                        if k == 1
                            S = computesymbols(jacobi, [1.0], θ)
                        else
                            S = computesymbols(
                                chebyshev,
                                [k, eigenvalue / 10.0, eigenvalue],
                                θ,
                            )
                        end
                        eigenvalues = [abs(val) for val in eigvals(S * M * S)]
                        currentmaxeigenvalue = max(eigenvalues...)
                        if (currentmaxeigenvalue > maxeigenvalue[k])
                            maxeigenvalue[k] = currentmaxeigenvalue
                            θ_maxeigenvalue[k, :] = θ / π
                        end
                    end
                end
            end
        end
        for k = 1:4
            append!(
                convergencefactors,
                DataFrame(
                    finep = finep,
                    coarsep = coarsep,
                    k = k,
                    θ = θ_maxeigenvalue[k, :],
                    ρ = maxeigenvalue[k],
                ),
            )
        end
    end
end

CSV.write("two-grid-chebyshev.csv", convergencefactors)
