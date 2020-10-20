# ------------------------------------------------------------------------------
# mass matrix example
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

# compute operator symbols
A = computesymbols(diffusion, [π, π])

# verify
eigenvalues = real(eigvals(A))
@test min(eigenvalues...) ≈ 0.24284729156204987
@test max(eigenvalues...) ≈ 7.852944737646179

# ------------------------------------------------------------------------------
