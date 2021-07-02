# ------------------------------------------------------------------------------
# linear elasticity example
# ------------------------------------------------------------------------------

using LinearAlgebra

# setup
mesh = Mesh3D(1.0, 1.0, 1.0)
finep = 2
coarsep = 1
numbercomponents = 3
dimension = 3
finebasis = TensorH1LagrangeBasis(finep + 1, finep + 1, numbercomponents, dimension)
coarsebasis = TensorH1LagrangeBasis(coarsep + 1, finep + 1, numbercomponents, dimension)
ctofbasis = TensorH1LagrangeBasis(
    coarsep + 1,
    finep + 1,
    numbercomponents,
    dimension,
    lagrangequadrature = true,
)

# constants
E = 1E6                             # Young's modulus
ν = 0.3                             # Poisson's ratio
λ = E * ν / ((1 + ν) * (1 - 2 * ν)) # Lamé parameters
μ = E / (2 * (1 + ν))

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
        OperatorField(basis, [EvaluationMode.quadratureweights], "quadrature weights"),
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

# Chebyshev smoother
chebyshev = Chebyshev(fineoperator)

# p-multigrid preconditioner
multigrid = PMultigrid(fineoperator, coarseoperator, chebyshev, [ctofbasis])

# compute operator symbols
A = computesymbols(multigrid, [3], [1, 1], [π, π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
