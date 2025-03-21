# ------------------------------------------------------------------------------
# hyperelasticity example
# ------------------------------------------------------------------------------

using LFAToolkit
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
    collocatedquadrature = true,
)

# constants
E = 1E6                             # Young's modulus
ν = 0.3                             # Poisson's ratio
K = E / (3 * (1 - 2 * ν))           # bulk modulus
λ = E * ν / ((1 + ν) * (1 - 2 * ν)) # Lamé parameters
μ = E / (2 * (1 + ν))

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
    dP = (dF * S + F * dS) * w[1]

    return [dP']
end

# linearized Neo-Hookean operators
begin
    local function makeoperator(basis::TensorBasis)
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
end

# Chebyshev smoother
chebyshev = Chebyshev(fineoperator)

# p-multigrid preconditioner
multigrid = PMultigrid(fineoperator, coarseoperator, chebyshev, [ctofbasis])

# compute operator symbols
A = computesymbols(multigrid, [3], [1, 1], [π, π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
