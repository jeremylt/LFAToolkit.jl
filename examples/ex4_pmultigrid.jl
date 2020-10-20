# ------------------------------------------------------------------------------
# p-multigrid example
# ------------------------------------------------------------------------------

# setup
mesh = Mesh2D(1.0, 1.0)
finebasis = TensorH1LagrangeBasis(4, 4, 2)
coarsebasis = TensorH1LagrangeBasis(2, 4, 2)
lagrangequadrature = true
ctofbasis = TensorH1LagrangeBasis(2, 4, 2, lagrangequadrature)

function diffusionweakform(du::Array{Float64}, w::Array{Float64})
    dv = du*w[1]
    return [dv]
end

# fine grid diffusion operator
fineinputs = [
    OperatorField(finebasis, [EvaluationMode.gradient]),
    OperatorField(finebasis, [EvaluationMode.quadratureweights]),
]
fineoutputs = [OperatorField(finebasis, [EvaluationMode.gradient])]
finediffusion = Operator(diffusionweakform, mesh, fineinputs, fineoutputs)

# coarse grid diffusion operator
coarseinputs = [
    OperatorField(coarsebasis, [EvaluationMode.gradient]),
    OperatorField(coarsebasis, [EvaluationMode.quadratureweights]),
]
coarseoutputs = [OperatorField(coarsebasis, [EvaluationMode.gradient])]
coarsediffusion = Operator(diffusionweakform, mesh, coarseinputs, coarseoutputs)

# Jacobi smoother
jacobi = Jacobi(finediffusion)

# p-multigrid preconditioner
multigrid = PMultigrid(finediffusion, coarsediffusion, jacobi, [ctofbasis])

# compute operator symbols
A = computesymbols(multigrid, [1.0], [1, 1], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
