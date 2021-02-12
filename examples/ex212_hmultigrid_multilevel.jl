# ------------------------------------------------------------------------------
# p-multigrid multilevel example
# ------------------------------------------------------------------------------

# setup
mesh = Mesh2D(1.0, 1.0)
p = 2
numberfineelements1d = 4
numbermidelements1d = 2
dimension = 2
ctombasis = TensorH1LagrangeHProlongationBasis(p, dimension, numbermidelements1d);
mtofbasis = TensorH1LagrangeHProlongationMacroBasis(
    p,
    dimension,
    numbermidelements1d,
    numberfineelements1d,
);

# operators
function diffusionweakform(du::Array{Float64}, w::Array{Float64})
    dv = du*w[1]
    return [dv]
end
# -- fine
basis = TensorH1LagrangeMacroBasis(p, p + 1, dimension, numberfineelements1d);
inputs = [
    OperatorField(basis, [EvaluationMode.gradient]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.gradient])];
finediffusion = Operator(diffusionweakform, mesh, inputs, outputs);
# -- mid
basis = TensorH1LagrangeMacroBasis(p, p + 1, dimension, numbermidelements1d);
inputs = [
    OperatorField(basis, [EvaluationMode.gradient]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.gradient])];
middiffusion = Operator(diffusionweakform, mesh, inputs, outputs);
# -- coarse
basis = TensorH1LagrangeBasis(p, p + 1, dimension);
inputs = [
    OperatorField(basis, [EvaluationMode.gradient]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.gradient])];
coarsediffusion = Operator(diffusionweakform, mesh, inputs, outputs);

# Jacobi smoothers
finejacobi = Jacobi(finediffusion)
midjacobi = Jacobi(middiffusion)

# h-multigrid preconditioner
midmultigrid = HMultigrid(middiffusion, coarsediffusion, midjacobi, [ctombasis])
multigrid = HMultigrid(finediffusion, midmultigrid, finejacobi, [mtofbasis])

# compute operator symbols
A = computesymbols(multigrid, [0.7], [1, 1], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
