# ------------------------------------------------------------------------------
# h-multigrid example
# ------------------------------------------------------------------------------

using LinearAlgebra

# setup
mesh = Mesh2D(1.0, 1.0)
p = 2
numberfineelements1d = 2
numbercomponents = 1
dimension = 2
ctofbasis =
    TensorH1LagrangeHProlongationBasis(p, numbercomponents, dimension, numberfineelements1d);

# operators
function diffusionweakform(du::Array{Float64}, w::Array{Float64})
    dv = du * w[1]
    return [dv]
end
# -- fine
basis =
    TensorH1LagrangeMacroBasis(p, p + 1, numbercomponents, dimension, numberfineelements1d);
inputs = [
    OperatorField(basis, [EvaluationMode.gradient]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.gradient])];
finediffusion = Operator(diffusionweakform, mesh, inputs, outputs);
# -- coarse
basis = TensorH1LagrangeBasis(p, p + 1, numbercomponents, dimension);
inputs = [
    OperatorField(basis, [EvaluationMode.gradient]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
];
outputs = [OperatorField(basis, [EvaluationMode.gradient])];
coarsediffusion = Operator(diffusionweakform, mesh, inputs, outputs);

# Chebyshev smoother
chebyshev = Chebyshev(finediffusion)

# h-multigrid preconditioner
multigrid = HMultigrid(finediffusion, coarsediffusion, chebyshev, [ctofbasis])

# compute operator symbols
A = computesymbols(multigrid, [3], [1, 1], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
