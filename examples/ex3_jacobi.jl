# ------------------------------------------------------------------------------
# Jacobi smoother example
# ------------------------------------------------------------------------------

# setup
mesh = Mesh2D(1.0, 1.0)
basis = TensorH1LagrangeBasis(4, 4, 2)

function diffusionweakform(du::Array{Float64}, w::Array{Float64})
    dv = du*w[1]
    return [dv]
end

# diffusion operator
inputs = [
    OperatorField(basis, [EvaluationMode.gradient]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
]
outputs = [OperatorField(basis, [EvaluationMode.gradient])]
diffusion = Operator(diffusionweakform, mesh, inputs, outputs)

# Jacobi smoother
jacobi = Jacobi(diffusion)

# compute operator symbols
A = computesymbols(jacobi, [1.0], [π, π])

# symbols eigenvalues
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
