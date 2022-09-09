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
#mapping = sausage_transformation(9)
#mapping = kosloff_talezer_transformation(0.98)
#mapping = hale_trefethen_strip_transformation(1.4)
basis =
    TensorH1LagrangeBasis(P, Q, 1, 1, collocatedquadrature=collocate, mapping=mapping)

# frequency set up
numbersteps = 100
θ_min = 0
θ_max = (P - 1) * π
θ = LinRange(θ_min, θ_max, numbersteps)

# associated phase speed
c = 1.0

# Tau scaling for SUPG
τ = 0.5 / (P - 1) # 0 returns Galerkin method; empirically, 0.25 is too big for P=3 (quadratic)

# weak form for SUPG advection
function supgadvectionweakform(U::Matrix{Float64}, w::Array{Float64})
    u = U[1, :]
    du = U[2, :]
    dv = (c * u - c * τ * (c * du)) * w[1]
    return [dv]
end

# weak form for SUPG mass matrix
# udot will be obtained from M udot  = A u
function supgmassweakform(udot::Array{Float64}, w::Array{Float64})
    v = udot * w[1]
    dv = c * τ * udot * w[1]
    return [v, dv]
end

# supg advection operator
inputs = [
    OperatorField(basis, [EvaluationMode.interpolation, EvaluationMode.gradient], "advected field"),
    OperatorField(basis, [EvaluationMode.quadratureweights], "quadrature weights"),
]
outputs = [OperatorField(basis, [EvaluationMode.gradient])]
#outputs = [OperatorField(basis, [EvaluationMode.interpolation, EvaluationMode.gradient,])]
supgadvection = Operator(supgadvectionweakform, mesh, inputs, outputs)

function supgmassweakform(udot::Array{Float64}, w::Array{Float64})
    v = udot * w[1]
    dv = c * τ * udot * w[1]
    return ([v; dv],)
end
mass = Operator(supgmassweakform, mesh,
    [OperatorField(basis, [EvaluationMode.interpolation], "u_t"),
        OperatorField(basis, [EvaluationMode.quadratureweights], "quadrature weights")],
    [OperatorField(basis, [EvaluationMode.interpolation, EvaluationMode.gradient])])

# compute operator symbols
function advection_supg_symbol(θ)
    A = computesymbols(supgadvection, [θ]) * 2 # transform from reference to physical on dx=1 grid
    M = computesymbols(mass, [θ]) # mass matrix
    return sort(imag.(eigvals(-M \ A)))
end

S = hcat(advection_supg_symbol.(θ)...)'
if true
    plot(θ / π, S ./ π, linewidth=3)  # Dispersion
    plot!(
        identity,
        xlabel="θ/π",
        ylabel="Eigenvalues",
        label="exact",
        legend=:none,
        color=:black,
    )
else
    plot(θ / π, S ./ θ, linewidth=3)    # Phase speed plot
    plot!(
        one,
        xlabel="θ/π",
        ylabel="Phase speed",
        legend=:none,
        color=:black,
        ylim=(0.95, 1.05),
    )
end
plot!(title="P=$P, collocate=$collocate, τ=$τ")
# ---------------------------------------------------------------
