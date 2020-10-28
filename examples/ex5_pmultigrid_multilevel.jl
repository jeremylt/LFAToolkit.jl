# ------------------------------------------------------------------------------
# p-multigrid multilevel example
# ------------------------------------------------------------------------------

# setup
mesh = Mesh2D(1.0, 1.0)
finebasis = TensorH1LagrangeBasis(5, 5, 2)
midbasis = TensorH1LagrangeBasis(3, 5, 2)
coarsebasis = TensorH1LagrangeBasis(2, 5, 2)
ctombasis = TensorH1LagrangeBasis(2, 3, 2, lagrangequadrature = true)
mtofbasis = TensorH1LagrangeBasis(3, 5, 2, lagrangequadrature = true)

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

# mid grid diffusion operator
midinputs = [
    OperatorField(midbasis, [EvaluationMode.gradient]),
    OperatorField(midbasis, [EvaluationMode.quadratureweights]),
]
midoutputs = [OperatorField(midbasis, [EvaluationMode.gradient])]
middiffusion = Operator(diffusionweakform, mesh, midinputs, midoutputs)

# coarse grid diffusion operator
coarseinputs = [
    OperatorField(coarsebasis, [EvaluationMode.gradient]),
    OperatorField(coarsebasis, [EvaluationMode.quadratureweights]),
]
coarseoutputs = [OperatorField(coarsebasis, [EvaluationMode.gradient])]
coarsediffusion = Operator(diffusionweakform, mesh, coarseinputs, coarseoutputs)

# Jacobi smoothers
finejacobi = Jacobi(finediffusion)
midjacobi = Jacobi(middiffusion)

# p-multigrid preconditioner
midmultigrid = PMultigrid(middiffusion, coarsediffusion, midjacobi, [ctombasis])
multigrid = PMultigrid(finediffusion, midmultigrid, finejacobi, [mtofbasis])

# compute operator symbols
A = computesymbols(multigrid, [0.651], [1, 1], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
