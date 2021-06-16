# dependencies
using LFAToolkit
using LinearAlgebra
using CSV
using DataFrames

# setup
dimension = 3
numbercomponents = 3
mesh = Mesh3D(1.0, 1.0, 1.0)
maxeigenvalues = DataFrame()

# constants
e = 1E6                             # Young's modulus
ν = 0.3                             # Poisson's ratio
K = e / (3 * (1 - 2 * ν))           # bulk modulus
λ = e * ν / ((1 + ν) * (1 - 2 * ν)) # Lamé parameters
μ = e / (2 * (1 + ν))

# test range
for fineP = 1:4
    println("  fine_p = ", 2^fineP)
    for coarseP = 0:fineP-1
        println("  coarse_p = ", 2^coarseP)
        # setup
        # -- bases
        coarsep = 2^coarseP
        finep = 2^fineP

        finebasis = TensorH1LagrangeBasis(finep, finep, numbercomponents, dimension)
        coarsebasis = TensorH1LagrangeBasis(coarsep, finep, numbercomponents, dimension)
        ctofbasis = TensorH1LagrangeBasis(
            coarsep,
            finep,
            numbercomponents,
            dimension,
            lagrangequadrature = true,
        )

        # state
        gradu = [1; 2; 3] * ones(1, 3);

        function neohookeanweakform(deltadu::Array{Float64}, w::Array{Float64})
            # dP = dF S + F dS

            # deformation gradient
            F = gradu + I
            J = det(F)
            # Green-Lagrange strain tensor
            E = (gradu * gradu' + gradu' * gradu) / 2
            # right Cauchy-Green tensor
            C = 2 * E + I
            C_inv = C^-1
            # second Piola-Kirchhoff
            S = λ * log(J) * C_inv + 2 * μ * C_inv * E

            # delta du
            deltadu = deltadu'
            # dF
            dF = deltadu + I
            # deltaE
            deltaE = (deltadu * deltadu' + deltadu' * deltadu) / 2
            # dS
            dS = λ * sum(C_inv .* deltaE) * C_inv + 2 * (μ - λ * log(J)) * C_inv * deltaE * C_inv
            # dP
            dP = dF * S + F * dS

            return [dP']
        end

        # linearized Neo-Hookean operators
        function makeoperator(basis::TensorBasis)
            inputs = [
                OperatorField(basis, [EvaluationMode.gradient], "gradent of deformation"),
                OperatorField(basis, [EvaluationMode.quadratureweights], "quadrature weights"),
            ]
            outputs = [
                OperatorField(
                    basis,
                    [EvaluationMode.gradient],
                    "test function gradient of deformation",
                ),
            ]
            return Operator(neohookeanweakform, mesh, inputs, outputs)
        end
        fineoperator = makeoperator(finebasis)
        coarseoperator = makeoperator(coarsebasis)

        # Chebyshev smoother
        chebyshev = Chebyshev(fineoperator)

        # p-multigrid preconditioner
        multigrid = PMultigrid(fineoperator, coarseoperator, chebyshev, [ctofbasis])

        # compute smoothing factor
        # -- setup
        numberruns = 100
        maxeigenvalue = 0
        θ_min = -π/2
        θ_max = 3π/2

        # -- compute
        for ω = 1:4
            println("    ω = ", ω)
            maxeigenvalue = 0
            ω_maxegenvalue = -1
            θ_maxegenvalue = -1
            for i in 1:numberruns, j in 1:numberruns, k in 1:numberruns
                θ = [
                    θ_min + (θ_max - θ_min)*i/numberruns,
                    θ_min + (θ_max - θ_min)*j/numberruns,
                    θ_min + (θ_max - θ_min)*k/numberruns
                ]
                if sqrt(abs(θ[1] % 2π)^2 + abs(θ[2] % 2π + abs(θ[3] % 2π)^2) > π/128
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
            append!(maxeigenvalues, DataFrame(finep=finep, coarsep=coarsep, ω=ω, θ=θ_maxegenvalue, rho=maxeigenvalue))
        end
    end
end

CSV.write("two-grid-hyperelasticity.csv", maxeigenvalues)
