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
gradu = zeros(3, 3);
gradu[1, 1:3] = 1*ones(3);
gradu[2, 1:3] = 2*ones(3);
gradu[3, 1:3] = 3*ones(3);

function neohookeanweakform(
    deltadux::Array{Float64},
    deltaduy::Array{Float64},
    deltaduz::Array{Float64},
    w::Array{Float64},
)
    # dP = dF S + F dS
    #  dS = partialS/partialE

    # deformation gradient
    F = gradu + I
    J = det(F)
    # Green-Lagrange strain tensor
    E = (gradu*transpose(gradu) + transpose(gradu)*gradu)/2
    # right Cauchy-Green tensor
    C = 2*E + I
    C_inv = C^-1
    # Second Piola-Kirchhoff
    S = λ*log(J)*C_inv + μ*(I - C_inv)

    # delta du
    deltadu = zeros(3, 3)
    deltadu[1, 1:3] = deltadux
    deltadu[2, 1:3] = deltaduy
    deltadu[3, 1:3] = deltaduz
    # dF
    dF = deltadu + I
    # deltaE
    deltaE = (deltadu*transpose(deltadu) + transpose(deltadu)*deltadu)/2
    # dS
    dS = λ*sum(C_inv.*deltaE)*C_inv + 2*(μ - λ*log(J))*C_inv*deltaE*C_inv
    # dP
    dP = dF*S + F*dS

    return [dP[1, 1:3], dP[2, 1:3], dP[3, 1:3]]
end

# diffusion operators
# -- fine level
inputs = [
    OperatorField(finebasis, [EvaluationMode.gradient]),
    OperatorField(finebasis, [EvaluationMode.gradient]),
    OperatorField(finebasis, [EvaluationMode.gradient]),
    OperatorField(finebasis, [EvaluationMode.quadratureweights]),
]
outputs = [
    OperatorField(finebasis, [EvaluationMode.gradient]),
    OperatorField(finebasis, [EvaluationMode.gradient]),
    OperatorField(finebasis, [EvaluationMode.gradient]),
]
fineoperator = Operator(neohookeanweakform, mesh, inputs, outputs)
# -- coarse level
inputs = [
    OperatorField(coarsebasis, [EvaluationMode.gradient]),
    OperatorField(coarsebasis, [EvaluationMode.gradient]),
    OperatorField(coarsebasis, [EvaluationMode.gradient]),
    OperatorField(coarsebasis, [EvaluationMode.quadratureweights]),
]
outputs = [
    OperatorField(coarsebasis, [EvaluationMode.gradient]),
    OperatorField(coarsebasis, [EvaluationMode.gradient]),
    OperatorField(coarsebasis, [EvaluationMode.gradient]),
]
coarseoperator = Operator(neohookeanweakform, mesh, inputs, outputs)

# Chebyshev smoother
chebyshev = Chebyshev(fineoperator)

# p-multigrid preconditioner
multigrid =
    PMultigrid(fineoperator, coarseoperator, chebyshev, [ctofbasis, ctofbasis, ctofbasis])

# compute operator symbols
A = computesymbols(multigrid, [4], [1, 1], [π, π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
