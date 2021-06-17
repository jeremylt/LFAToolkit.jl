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

        # -- Chebyshev smoother
        chebyshev = Chebyshev(finediffusion)

        # -- p-multigrid preconditioner
        multigrid = PMultigrid(finediffusion, coarsediffusion, chebyshev, [ctofbasis])

        # compute smoothing factor
        # -- setup
        numberruns = 100
        maxeigenvalue = 0
        θ_min = -π / 2
        θ_max = 3π / 2

        # -- compute
        for ω = 1:4
            println("    ω = ", ω)
            maxeigenvalue = 0
            ω_maxegenvalue = -1
            θ_maxegenvalue = -1
            # -- 1D --
            if dimension == 1
                for i = 1:numberruns
                    θ = [θ_min + (θ_max - θ_min) * i / numberruns]
                    if abs(θ[1] % 2π) > π / 128
                        A = computesymbols(multigrid, [ω], [1, 1], θ)
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
                    if sqrt(abs(θ[1] % 2π)^2 + abs(θ[2] % 2π)^2) > π / 128
                        A = computesymbols(multigrid, [ω], [1, 1], θ)
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
            append!(
                maxeigenvalues,
                DataFrame(
                    finep = finep,
                    coarsep = coarsep,
                    ω = ω,
                    θ = θ_maxegenvalue,
                    ρ = maxeigenvalue,
                ),
            )
        end
    end
end

CSV.write("two-grid-chebyshev.csv", maxeigenvalues)
