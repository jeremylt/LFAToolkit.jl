# dependencies
using LFAToolkit
using LinearAlgebra
using CSV
using DataFrames

# setup
dimension = 2
numbercomponents = 1
mesh = []
if dimension == 1
    mesh = Mesh1D(1.0)
elseif dimension == 2
    mesh = Mesh2D(1.0, 1.0)
end
convergencefactors = DataFrame()

lowerscaling = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
numlowerscaling = 7

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
        numbersteps = 32
        θ_min = -π / 2
        θ_max = 3π / 2
        θ_step = 2π / numbersteps
        θ_range = θ_min:θ_step:(θ_max-θ_step)

        # -- compute
        maxeigenvalue = zeros(numlowerscaling, 4)
        θ_maxeigenvalue = -1 * ones(numlowerscaling, 4, dimension)
        # -- 1D --
        if dimension == 1
            for i = 1:numbersteps
                θ = [θ_range[i]]
                if abs(θ[1]) > π / 128
                    M = computesymbols(multigrid, [0], [0, 0], θ)
                    for k = 1:4, l = 1:numlowerscaling
                        seteigenvalueestimatescaling(chebyshev, [0.0, lowerscaling[l], 0.0, 1.0])
                        S = computesymbols(chebyshev, [k], θ)
                        eigenvalues = [abs(val) for val in eigvals(S * M * S)]
                        currentmaxeigenvalue = max(eigenvalues...)
                        if (currentmaxeigenvalue > maxeigenvalue[l, k])
                            maxeigenvalue[l, k] = currentmaxeigenvalue
                            θ_maxeigenvalue[l, k, :] = θ / π
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
                    for k = 1:4, l = 1:numlowerscaling
                        seteigenvalueestimatescaling(chebyshev, [0.0, lowerscaling[l], 0.0, 1.0])
                        S = computesymbols(chebyshev, [k], θ)
                        eigenvalues = [abs(val) for val in eigvals(S * M * S)]
                        currentmaxeigenvalue = max(eigenvalues...)
                        if (currentmaxeigenvalue > maxeigenvalue[l, k])
                            maxeigenvalue[l, k] = currentmaxeigenvalue
                            θ_maxeigenvalue[l, k, :] = θ / π
                        end
                    end
                end
            end
        end
        for k = 1:4, l = 1:numlowerscaling
            append!(
                convergencefactors,
                DataFrame(
                    finep = finep,
                    coarsep = coarsep,
                    k = k,
                    c = lowerscaling[l],
                    θ = θ_maxeigenvalue[l, k, :],
                    ρ = maxeigenvalue[l, k],
                ),
            )
        end
    end
end

CSV.write("two-grid-chebyshev-eigenvalues.csv", convergencefactors)
