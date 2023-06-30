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

# setup
Δx = 1.0
mesh = Mesh1D(Δx)
dxdξ = Δx / 2 # 2 comes from quadrature domain of [-1,1]
dξdx = 1 / dxdξ
det_dxdξ = dxdξ # Determinant of mapping Jacobian

p = 2;
q = p;
collocate = false
mapping = nothing
basis =
    TensorH1LagrangeBasis(p, q, 1, 1, collocatedquadrature = collocate, mapping = mapping)

# frequency set up
numbersteps = 100
θ_min = 0
θ_max = (p - 1) * π
θ = LinRange(θ_min, θ_max, numbersteps)

# associated phase speed
c = 1.0

# Tau scaling for SUPG
τ = 0.5 / (p - 1) # 0 returns Galerkin method; empirically, 0.25 is too big for p = 3 (quadratic)

# weak form for SUPG advection
function supgadvectionweakform(U::Matrix{Float64}, w::Array{Float64})
    u = U[1, :]
    du = U[2, :]
    dv = dξdx * (c * u - c * τ * (c * du * dξdx)) * w[1] * det_dxdξ
    return [dv]
end

# supg advection operator
inputs = [
    OperatorField(
        basis,
        [EvaluationMode.interpolation, EvaluationMode.gradient],
        "advected field",
    ),
    OperatorField(basis, [EvaluationMode.quadratureweights], "quadrature weights"),
]
outputs = [OperatorField(basis, [EvaluationMode.gradient])]

supgadvection = Operator(supgadvectionweakform, mesh, inputs, outputs)

# supg mass operator
function supgmassweakform(udot::Array{Float64}, w::Array{Float64})
    v = udot * w[1] * det_dxdξ
    dv = dξdx * c * τ * udot * w[1] * det_dxdξ
    return ([v; dv],)
end
supgmass = Operator(
    supgmassweakform,
    mesh,
    [
        OperatorField(basis, [EvaluationMode.interpolation], "uₜ"),
        OperatorField(basis, [EvaluationMode.quadratureweights], "quadrature weights"),
    ],
    [OperatorField(basis, [EvaluationMode.interpolation, EvaluationMode.gradient])],
)

# compute operator symbols
function advection_supg_symbol(θ)
    A = computesymbols(supgadvection, [θ]) # transform from reference to physical on dx=1 grid
    M = computesymbols(supgmass, [θ]) # mass matrix
    return sort(imag.(eigvals(-M \ A)))
end

eigenvalues = hcat(advection_supg_symbol.(θ)...)'

# ---------------------------------------------------------------
