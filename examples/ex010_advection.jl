# ---------------------------------------------------------------
# advection example following Melvin, Staniforth and Thuburn
# Q.J:R. Meteorol. Soc. 138: 1934-1947, Oct (2012)
# ---------------------------------------------------------------
using LFAToolkit
using LinearAlgebra
using Plots

function kosloff_tal_ezer(α)
    g(s) = asin(α * s) / asin(α)
    g_prime(s) = α / (asin(α) * sqrt(1 - (α * s)^2))
    g, g_prime
end

function hale_trefethen_strip(ρ)
    τ = π / log(ρ)
    d = .5 + 1 / (exp(τ * π) + 1)
    π2 = π / 2
    # Unscaled functions of u
    g_u(u) = log(1 + exp(-τ*(π2 + u))) - log(1 + exp(-τ*(π2-u))) + d*τ*u
    g_prime_u(u) = 1 / (exp(τ*(π2+u)) + 1) + 1 / (exp(τ*(π2-u)) + 1) -d
    # Normalizing factor and scaled functions of s
    C = 1 / g_u(π/2)
    g(s) = C * g_u(asin(s))
    g_prime(s) = -τ*C/sqrt(1 - s^2) * g_prime_u(asin(s))
    g, g_prime
end

# setup
mesh = Mesh1D(1.0)
P = 11;
Q = P;
collocate = false
#mapping = kosloff_tal_ezer(.98)
#mapping = sausage(29)
mapping = hale_trefethen_strip(1.4)
basis = TensorH1LagrangeBasis(P, Q, 1, 1, collocatedquadrature = collocate, mapping = mapping)

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
mass = GalleryOperator("mass", P, Q, mesh, collocatedquadrature = collocate, mapping = mapping)

# compute operator symbols
function advection_symbol(θ)
    A = computesymbols(advection, [θ]) * 2 # transform from reference to physical on dx=1 grid
    M = computesymbols(mass, [θ])
    return sort(imag.(eigvals(-M \ A)))
end

# compute transformation matrix 
function transformation_matrix(θ)
    R = computewavenumbertransformation(advection, [θ])
    return imag.(eigvals(R))
end

#θ = LinRange(.01, π, 20)
numbersteps = 250
θ_min = 0
θ_max = (P - 1) * π
θ = LinRange(θ_min, θ_max, numbersteps)

S = hcat(advection_symbol.(θ)...)'
#min_S = minimum(S)
#max_S = maximum(S)
plot(θ / π, S ./ π)
plot!(identity, label="exact", legend=:none, color=:black, ylims=(0, P+2))

#tm = hcat(transformation_matrix.(θ)...)'
#@show tm

# ---------------------------------------------------------------
