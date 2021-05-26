# ------------------------------------------------------------------------------
# lumped BDDC on macro elements example
# ------------------------------------------------------------------------------

using LinearAlgebra

# setup
mesh = Mesh2D(1.0, 1.0)
p = 2
numberfineelements1d = 4
numbercomponents = 1
dimension = 2

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

# BDDC smoother
bddc = DirichletBDDC(finediffusion)

# compute operator symbols
A = computesymbols(bddc, [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
