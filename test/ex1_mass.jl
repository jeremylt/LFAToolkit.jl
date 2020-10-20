# ------------------------------------------------------------------------------
# mass matrix example
# ------------------------------------------------------------------------------

# setup
mesh = Mesh2D(1.0, 1.0)
basis = TensorH1LagrangeBasis(4, 4, 2)

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

# verify
eigenvalues = real(eigvals(A))
@test min(eigenvalues...) ≈ 0.008379422444571976
@test max(eigenvalues...) ≈ 0.17361111111111088

# ------------------------------------------------------------------------------
