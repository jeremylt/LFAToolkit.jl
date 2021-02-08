# ------------------------------------------------------------------------------
# solid mechanics example
# ------------------------------------------------------------------------------

# setup
mesh = Mesh3D(1.0, 1.0, 1.0)
finep = 3
coarsep = 2
dimension = 3
finebasis = TensorH1LagrangeBasis(finep, finep, dimension)
coarsebasis = TensorH1LagrangeBasis(coarsep, finep, dimension)
ctofbasis = TensorH1LagrangeBasis(coarsep, finep, dimension, lagrangequadrature = true)

function neohookeanweakform(du::Array{Float64}, w::Array{Float64})
    dv = du*w[1]
    return [dv]
end

# diffusion operators
# -- fine level
inputs = [
    OperatorField(finebasis, [EvaluationMode.gradient]),
    OperatorField(finebasis, [EvaluationMode.quadratureweights]),
]
outputs = [OperatorField(finebasis, [EvaluationMode.gradient])]
fineoperator = Operator(neohookeanweakform, mesh, inputs, outputs)
# -- coarse level
inputs = [
    OperatorField(coarsebasis, [EvaluationMode.gradient]),
    OperatorField(coarsebasis, [EvaluationMode.quadratureweights]),
]
outputs = [OperatorField(coarsebasis, [EvaluationMode.gradient])]
coarseoperator = Operator(neohookeanweakform, mesh, inputs, outputs)

# Chebyshev smoother
chebyshev = Chebyshev(fineoperator)

# p-multigrid preconditioner
multigrid = PMultigrid(fineoperator, coarseoperator, chebyshev, [ctofbasis])

# compute operator symbols
A = computesymbols(multigrid, [3], [1, 1], [π, π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
