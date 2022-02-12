# ---------------------------------------------------------------
# non polynomial advection mapped example using transplanted
# quadrature formulas from conformal maps following Hale and 
# Trefethen (2008) SIAM J. NUMER. ANAL. Vol. 46, No. 2
# ---------------------------------------------------------------
using LFAToolkit
using LinearAlgebra
using Plots

# setup
mesh = Mesh1D(1.0)
P = 3
Q = P
basis = TensorH1LagrangeBasis(P, Q, 1, 1)

# associated phase speed
h0 = 1.0
g = 9.81
U = sqrt(g * h0)

# weak form
function advectionmappedweakform(u::Array{Float64}, w::Array{Float64})
    dv = U * u * w[1]
    return [dv]
end

# mapped advection operator
# multiply differentiation basis matrix by inverse of g' 
# multiply quadrature weights by g'
inputs = [
    OperatorField(basis, [EvaluationMode.interpolation]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
]
outputs = [OperatorField(basis, [EvaluationMode.gradient])]
advectionmapped = Operator(advectionmappedweakform, mesh, inputs, outputs)
mass = GalleryOperator("mass", P, Q, mesh)

# compute operator symbols
A = computesymbols(advectionmapped, [π])
M = computesymbols(mass, [π])
eigenvalues = real(eigvals(-M \ A))

# ---------------------------------------------------------------
