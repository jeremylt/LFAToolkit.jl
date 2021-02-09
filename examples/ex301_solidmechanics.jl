# ------------------------------------------------------------------------------
# solid mechanics example
# ------------------------------------------------------------------------------

using LinearAlgebra

# setup
mesh = Mesh3D(1.0, 1.0, 1.0)
finep = 3
coarsep = 2
dimension = 3
finebasis = TensorH1LagrangeBasis(finep, finep, dimension)
coarsebasis = TensorH1LagrangeBasis(coarsep, finep, dimension)
ctofbasis = TensorH1LagrangeBasis(coarsep, finep, dimension, lagrangequadrature = true)

# constants
e = 1E6                     # Young's modulus
ν = 0.3                     # Poisson's ratio
K = e/(3*(1 - 2*ν))         # bulk modulus
λ = e*ν/((1 + ν)*(1 - 2*ν)) # Lamé parameters
μ = e/(2*(1 + ν))

# state
gradu = [1; 2; 3] * ones(1, 3);

function neohookeanweakform(
    deltadux::Array{Float64},
    deltaduy::Array{Float64},
    deltaduz::Array{Float64},
    w::Array{Float64},
)
    # dP = dF S + F dS

    # deformation gradient
    F = gradu + I
    J = det(F)
    # Green-Lagrange strain tensor
    E = (gradu*gradu' + gradu'*gradu)/2
    # right Cauchy-Green tensor
    C = 2*E + I
    C_inv = C^-1
    # second Piola-Kirchhoff
    S = λ*log(J)*C_inv + 2*μ*C_inv*E

    # delta du
    deltadu = [
        deltadux';
        deltaduy';
        deltaduz';
    ]
    # dF
    dF = deltadu + I
    # deltaE
    deltaE = (deltadu*deltadu' + deltadu'*deltadu)/2
    # dS
    dS = λ*sum(C_inv.*deltaE)*C_inv + 2*(μ - λ*log(J))*C_inv*deltaE*C_inv
    # dP
    dP = dF*S + F*dS

    return [dP[1, 1:3], dP[2, 1:3], dP[3, 1:3]]
end

# linearized Neo-Hookean operators
function makeoperator(basis::TensorBasis)
    inputs = [
        OperatorField(basis, [EvaluationMode.gradient], "gradent of deformation in x"),
        OperatorField(basis, [EvaluationMode.gradient], "gradent of deformation in y"),
        OperatorField(basis, [EvaluationMode.gradient], "gradent of deformation in z"),
        OperatorField(basis, [EvaluationMode.quadratureweights], "quadrature weights"),
    ]
    outputs = [
        OperatorField(
            basis,
            [EvaluationMode.gradient],
            "test function gradient of deformation in x",
        ),
        OperatorField(
            basis,
            [EvaluationMode.gradient],
            "test function gradient of deformation in y",
        ),
        OperatorField(
            basis,
            [EvaluationMode.gradient],
            "test function gradient of deformation in z",
        ),
    ]
    return Operator(neohookeanweakform, mesh, inputs, outputs)
end
fineoperator = makeoperator(finebasis)
coarseoperator = makeoperator(coarsebasis)

# Chebyshev smoother
chebyshev = Chebyshev(fineoperator)

# p-multigrid preconditioner
multigrid =
    PMultigrid(fineoperator, coarseoperator, chebyshev, [ctofbasis, ctofbasis, ctofbasis])

# compute operator symbols
A = computesymbols(multigrid, [4], [1, 1], [π, π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
