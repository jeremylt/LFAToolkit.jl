# dependencies
using LFAToolkit
using LinearAlgebra
using CSV
using DataFrames

# setup
dimension = 3
numbercomponents = 3
mesh = Mesh3D(1.0, 1.0, 1.0)
convergencefactors = DataFrame()

# test range
for fineP = 1:3
    println("fine_p = ", 2^fineP)
    for coarseP = 0:fineP-1
        println("  coarse_p = ", 2^coarseP)
        # setup
        # -- bases
        coarsep = 2^coarseP
        finep = 2^fineP
        finebasis = TensorH1LagrangeBasis(finep + 1, finep + 1, numbercomponents, dimension)
        coarsebasis =
            TensorH1LagrangeBasis(coarsep + 1, finep + 1, numbercomponents, dimension)
        ctofbasis = TensorH1LagrangeBasis(
            coarsep + 1,
            finep + 1,
            numbercomponents,
            dimension,
            lagrangequadrature = true,
        )

        # constants
        e = 1E6                             # Young's modulus
        ν = 0.3                             # Poisson's ratio
        λ = e * ν / ((1 + ν) * (1 - 2 * ν)) # Lamé parameters
        μ = e / (2 * (1 + ν))

        function linearelasticityweakform(deltadu::Array{Float64}, w::Array{Float64})
            # strain
            dϵ = (deltadu + deltadu') / 2
            # strain energy
            dσ_11 = (λ + 2μ) * dϵ[1, 1] + λ * dϵ[2, 2] + λ * dϵ[3, 3]
            dσ_22 = λ * dϵ[1, 1] + (λ + 2μ) * dϵ[2, 2] + λ * dϵ[3, 3]
            dσ_33 = λ * dϵ[1, 1] + λ * dϵ[2, 2] + (λ + 2μ) * dϵ[3, 3]
            dσ_12 = μ * dϵ[1, 2]
            dσ_13 = μ * dϵ[1, 3]
            dσ_23 = μ * dϵ[2, 3]
            dσ = [dσ_11 dσ_12 dσ_13; dσ_12 dσ_22 dσ_23; dσ_13 dσ_23 dσ_33]

            # delta dv
            deltadv = dσ * w[1]

            return [deltadv']
        end

        # linear elasticity operators
        function makeoperator(basis::TensorBasis)
            inputs = [
                OperatorField(basis, [EvaluationMode.gradient], "gradent of deformation"),
                OperatorField(
                    basis,
                    [EvaluationMode.quadratureweights],
                    "quadrature weights",
                ),
            ]
            outputs = [
                OperatorField(
                    basis,
                    [EvaluationMode.gradient],
                    "test function gradient of deformation",
                ),
            ]
            return Operator(linearelasticityweakform, mesh, inputs, outputs)
        end
        fineoperator = makeoperator(finebasis)
        coarseoperator = makeoperator(coarsebasis)

        # -- smoothers
        identity = IdentityPC(fineoperator)
        chebyshev = Chebyshev(fineoperator)

        # -- p-multigrid preconditioner
        multigrid = PMultigrid(fineoperator, coarseoperator, identity, [ctofbasis])

        # compute smoothing factor
        # -- setup
        numbersteps = 8
        θ_min = -π / 2
        θ_max = 3π / 2
        θ_step = 2π / numbersteps
        θ_range = θ_min:θ_step:(θ_max-θ_step)

        # -- compute
        maxeigenvalue = zeros(4)
        θ_maxeigenvalue = -1 * ones(4, dimension)
        for i = 1:numbersteps, j = 1:numbersteps, k = 1:numbersteps
            θ = [θ_range[i], θ_range[j], θ_range[k]]
            if sqrt(abs(θ[1])^2 + abs(θ[2])^2 + abs(θ[3])^2) > π / 128
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

CSV.write("two-grid-linear-elasticity.csv", convergencefactors)
