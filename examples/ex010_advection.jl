# ---------------------------------------------------------------
# advection example following Melvin, Staniforth and Thuburn
# Q.J:R. Meteorol. Soc. 138: 1934-1947, Oct (2012)
# ---------------------------------------------------------------
using LFAToolkit
using LinearAlgebra
using Plots

# setup
mesh = Mesh1D(1.0)
P = 3;
Q = P;
basis = TensorH1LagrangeBasis(P, Q, 1, 1)

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
#advection = GalleryOperator("advection", P, Q, mesh)
mass = GalleryOperator("mass", P, Q, mesh)

# compute operator symbols
function advection_symbol(θ)
    A = computesymbols(advection, [θ])
    M = computesymbols(mass, [θ]) * 0.5 # transform from reference to physical on dx=1 grid
    return sort(imag.(eigvals(-M \ A)))
end

# compute transformation matrix 
function transformation_matrix(θ)
    R = computewavenumbertransformation(advection, [θ])
    return imag.(eigvals(R))
end

#θ = LinRange(.01, π, 20)
numbersteps = 250
θ_min = -π / 2
θ_max = 3π / 2
θ_step = 2π / (numbersteps - 1)
θ = θ_min:θ_step:θ_max

S = hcat(advection_symbol.(θ)...)'
#min_S = minimum(S)
#max_S = maximum(S)
plot(θ / π, S)

#tm = hcat(transformation_matrix.(θ)...)'
#@show tm

# ---------------------------------------------------------------
