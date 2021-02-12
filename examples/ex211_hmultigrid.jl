# ------------------------------------------------------------------------------
# h-multigrid example
# ------------------------------------------------------------------------------

# setup
mesh = Mesh2D(1.0, 1.0)
p = 2
numberfineelements1d = 2
dimension = 2
ctofbasis = TensorH1LagrangeHProlongationBasis(p, dimension, numberfineelements1d);

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
# -- coarse
basis = TensorH1LagrangeBasis(p, p + 1, dimension);
inputs = [
    OperatorField(basis, [EvaluationMode.gradient]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.gradient])];
coarsediffusion = Operator(diffusionweakform, mesh, inputs, outputs);

# Jacobi smoother
jacobi = Jacobi(finediffusion)

# h-multigrid preconditioner
multigrid = HMultigrid(finediffusion, coarsediffusion, jacobi, [ctofbasis])

# compute operator symbols
A = computesymbols(multigrid, [0.7], [1, 1], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
