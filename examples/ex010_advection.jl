# ---------------------------------------------------------------
# advection example following Melvin, Staniforth and Thuburn
# Q.J:R. Meteorol. Soc. 138: 1934-1947, Oct (2012)
# non polynomial advection example using transplanted
# quadrature formulas from conformal maps following Hale and 
# Trefethen (2008) SIAM J. NUMER. ANAL. Vol. 46, No. 2
# ---------------------------------------------------------------
using LFAToolkit
using LinearAlgebra
using Plots

# setup
mesh = Mesh1D(1.0)
P = 4;
Q = P;
collocate = false
mapping = sausage(29)
#mapping = kosloff_tal_ezer(0.98)
#mapping = hale_trefethen_strip(1.4)
basis = TensorH1LagrangeBasis(P, Q, 1, 1, collocatedquadrature = collocate, mapping = mapping)

# frequency set up
numbersteps = 100
θ_min = 0
θ_max = (P - 1) * π
θ = LinRange(θ_min, θ_max, numbersteps)

# associated phase speed
U = 1.0

# weak form
function advectionweakform(u::Array{Float64}, w::Array{Float64})
    dv = U * u * w[1]
    return [dv]
end

# advection operator
inputs = [
    OperatorField(basis, [EvaluationMode.interpolation], "advected field"),
    OperatorField(basis, [EvaluationMode.quadratureweights], "quadrature weights"),
]
outputs = [OperatorField(basis, [EvaluationMode.gradient])]
advection = Operator(advectionweakform, mesh, inputs, outputs)
mass = GalleryOperator("mass", P, Q, mesh, collocatedquadrature = collocate, mapping = mapping)
combined_operator = Operator(combinedoperator, mesh, inputs, outputs) # combined operator M^{-1}A 

# compute combined operator M^{-1}A symbols
function combinedoperator(θ::Array{Float64})
    A = computesymbols(advection, [θ]) * 2 # transform from reference to physical on dx=1 grid
    M = computesymbols(mass, [θ]) # mass matrix
    inv_MA = computesymbols(combined_operator, [θ]) # M^{-1}A
    return inv_MA
end

# compute operator symbols
function advection_symbol(θ)
    A = computesymbols(advection, [θ]) * 2 # transform from reference to physical on dx=1 grid
    M = computesymbols(mass, [θ]) # mass matrix
    return sort(imag.(eigvals(-M \ A)))
end

# compute transformation matrix 
function transformation_matrix(θ)
    R = computewavenumbertransformation(combined_operator, [θ])
    return imag.(eigvals(R))
end

Tm = hcat(transformation_matrix.(θ)...)'
plot(θ / π, Tm ./ π)
# ---------------------------------------------------------------
