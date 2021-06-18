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
        chebyshev = Chebyshev(finediffusion)

        # -- p-multigrid preconditioner
        multigrid = PMultigrid(finediffusion, coarsediffusion, identity, [ctofbasis])

        # compute smoothing factor
        # -- setup
        numberruns = 100
        θ_min = -π / 2
        θ_max = 3π / 2

        # -- compute
        maxeigenvalue = zeros(4)
        θ_maxegenvalue = -1 * ones(4, dimension)
        # -- 1D --
        if dimension == 1
            for i = 1:numberruns
                θ = [θ_min + (θ_max - θ_min) * i / numberruns]
                if abs(θ[1]) > π / 128
                    M = computesymbols(multigrid, [0], [0, 0], θ)
                    for k = 1:4
                        S = computesymbols(chebyshev, [k], θ)
                        eigenvalues = [abs(val) for val in eigvals(S * M * S)]
                        currentmaxeigenvalue = max(eigenvalues...)
                        if (currentmaxeigenvalue > maxeigenvalue[k])
                            maxeigenvalue[k] = currentmaxeigenvalue
                            θ_maxegenvalue[k, :] = θ / π
                        end
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
                if sqrt(abs(θ[1])^2 + abs(θ[2])^2) > π / 128
                    M = computesymbols(multigrid, [0], [0, 0], θ)
                    for k = 1:4
                        S = computesymbols(chebyshev, [k], θ)
                        eigenvalues = [abs(val) for val in eigvals(S * M * S)]
                        currentmaxeigenvalue = max(eigenvalues...)
                        if (currentmaxeigenvalue > maxeigenvalue[k])
                            maxeigenvalue[k] = currentmaxeigenvalue
                            θ_maxegenvalue[k, :] = θ / π
                        end
                    end
                end
            end
            # -- 3D --
        elseif dimension == 3
            for i = 1:numberruns, j = 1:numberruns, k = 1:numberruns
                θ = [
                    θ_min + (θ_max - θ_min) * i / numberruns,
                    θ_min + (θ_max - θ_min) * j / numberruns,
                    θ_min + (θ_max - θ_min) * k / numberruns,
                ]
                if sqrt(abs(θ[1])^2 + abs(θ[2])^2 + abs(θ[3])^2) > π / 128
                    M = computesymbols(multigrid, [0], [0, 0], θ)
                    for k = 1:4
                        eigenvalueestimates = [0.0, 1.3330, 1.9893, 1.6202, 2.2932, 1.8643]
                        eigenvalue = eigenvalueestimates[finep]
                        S = computesymbols(chebyshev, [k, eigenvalue / 10.0, eigenvalue], θ)
                        eigenvalues = [abs(val) for val in eigvals(S * M * S)]
                        currentmaxeigenvalue = max(eigenvalues...)
                        if (currentmaxeigenvalue > maxeigenvalue[k])
                            maxeigenvalue[k] = currentmaxeigenvalue
                            θ_maxegenvalue[k, :] = θ / π
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
                    θ = θ_maxegenvalue[k, :],
                    ρ = maxeigenvalue[k],
                ),
            )
        end
    end
end

CSV.write("two-grid-chebyshev.csv", convergencefactors)