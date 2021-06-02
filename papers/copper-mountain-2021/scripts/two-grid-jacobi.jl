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

        ctofbasis = TensorH1LagrangePProlongationBasis(coarsep+1, finep+1, numbercomponents, dimension)

        # -- diffusion operators
        finediffusion = GalleryOperator("diffusion", finep+1, finep+1, mesh)

        coarsediffusion = GalleryOperator("diffusion", coarsep+1, finep+1, mesh)

        # -- Jacobi smoother
        jacobi = Jacobi(finediffusion)

        # -- p-multigrid preconditioner
        multigrid = PMultigrid(finediffusion, coarsediffusion, jacobi, [ctofbasis])

        # compute smoothing factor
        # -- setup
        numberruns = 100
        maxeigenvalue = 0
        θ_min = -π/2
        θ_max = 3π/2

        # -- compute
        for v = 1:3
            println("    v = ", v)
            mineigenvalue = 1.0
            ω_minegenvalue = -1
            θ_minegenvalue = -1
            for ω = 0.50:0.01:1.05
                maxeigenvalue = 0
                ω_maxegenvalue = -1
                θ_maxegenvalue = -1
                # -- 1D --
                if dimension == 1
                    for i in 1:numberruns
                       θ = [θ_min + (θ_max - θ_min)*i/numberruns]
                        if abs(θ[1] % 2π) > π/128
                            A = computesymbols(multigrid, [ω], [v, v], θ)
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
                    for i in 1:numberruns, j in 1:numberruns
                        θ = [
                            θ_min + (θ_max - θ_min)*i/numberruns,
                            θ_min + (θ_max - θ_min)*j/numberruns
                        ]
                        if sqrt(abs(θ[1] % 2π)^2 + abs(θ[2] % 2π)^2) > π/128
                            A = computesymbols(multigrid, [ω], [v, v], θ)
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
                     θ_minegenvalue = θ_maxegenvalue/π
                end
            end
            append!(maxeigenvalues, DataFrame(finep=finep, coarsep=coarsep, ω=ω_minegenvalue, θ=θ_minegenvalue, v=v, rho=mineigenvalue))
        end
    end
end

CSV.write("two-grid-jacobi.csv", maxeigenvalues)
