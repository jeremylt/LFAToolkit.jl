# ------------------------------------------------------------------------------
# mass matrix example
# ------------------------------------------------------------------------------

using LinearAlgebra

# setup
mesh = Mesh2D(1.0, 1.0)
basis = TensorH1LagrangeBasis(4, 4, 1, 2)

# weak form
function massweakform(u::Array{Float64}, w::Array{Float64})
    v = u*w[1]
    return [v]
end

# mass operator
inputs = [
    OperatorField(basis, [EvaluationMode.interpolation]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
]
outputs = [OperatorField(basis, [EvaluationMode.interpolation])]
mass = Operator(massweakform, mesh, inputs, outputs)

# compute operator symbols
A = computesymbols(mass, [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
