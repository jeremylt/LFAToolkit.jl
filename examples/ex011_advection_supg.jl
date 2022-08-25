# ---------------------------------------------------------------
# SUPG advection example following Melvin, Staniforth and Thuburn
# Q.J:R. Meteorol. Soc. 138: 1934-1947, Oct (2012)
# non polynomial advection example using transplanted (transformed)
# quadrature formulas from conformal maps following Hale and
# Trefethen (2008) SIAM J. NUMER. ANAL. Vol. 46, No. 2
# In this example, we can call other mapping options available
# i.e, sausage_transformation(9), kosloff_talezer_transformation(0.98)
# otherwise, run without mapping
# ---------------------------------------------------------------
using LFAToolkit
using LinearAlgebra
using Plots

# setup
mesh = Mesh1D(1.0)
P = 2;
Q = P;
collocate = false
mapping = nothing
#mapping = sausage(29)
#mapping = kosloff_tal_ezer(0.98)
#mapping = hale_trefethen_strip(1.4)
basis =
    TensorH1LagrangeBasis(P, Q, 1, 1, collocatedquadrature = collocate, mapping = mapping)

# frequency set up
numbersteps = 100
θ_min = 0
θ_max = (P - 1) * π
θ = LinRange(θ_min, θ_max, numbersteps)

# associated phase speed
c = 1.0

# Tau scaling for SUPG
τ = 1.0 # 0 returns Galerkin method

# weak form for SUPG advection
function supgadvectionweakform(u::Array{Float64}, du::Array{Float64}, w::Array{Float64})
    dv = (c * u - c * τ * (c * du)) * w[1]
    return [dv]
end

# supg advection operator
inputs = [
    OperatorField(basis, [EvaluationMode.interpolation], "advected field"),
    OperatorField(basis, [EvaluationMode.gradient], "gradient field"),
    OperatorField(basis, [EvaluationMode.quadratureweights], "quadrature weights"),
]
outputs = [OperatorField(basis, [EvaluationMode.gradient])]
supgadvection = Operator(supgadvectionweakform, mesh, inputs, outputs)
mass =
    GalleryOperator("mass", P, Q, mesh, collocatedquadrature = collocate, mapping = mapping)

# compute operator symbols
function advection_supg_symbol(θ)
    A = computesymbols(supgadvection, [θ]) * 2 # transform from reference to physical on dx=1 grid
    M = computesymbols(mass, [θ]) # mass matrix
    return sort(imag.(eigvals(-M \ A)))
end

S = hcat(advection_supg_symbol.(θ)...)'
#plot(θ / π, (S./θ)./ π , linewidth=3)    # Phase speed plot
plot(θ / π, S ./ π, linewidth = 3)
plot!(
    identity,
    xlabel = "θ/π",
    ylabel = "Eigenvalues",
    label = "exact",
    linewidth = 3,
    legend = :none,
    color = :black,
    ylims = (0, P),
)
# ---------------------------------------------------------------
