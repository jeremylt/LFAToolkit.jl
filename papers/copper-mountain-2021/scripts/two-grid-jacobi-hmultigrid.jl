# dependencies
using LFAToolkit
using LinearAlgebra
using CSV
using DataFrames

# setup
maxeigenvalues = DataFrame()

struct RunOption
    p::Int
    ω::Float64
    v::AbstractArray{Int,1}
    dimension::Int
end

runoptions = [
    RunOption(2, 1.000, [0, 1], 1),
    RunOption(2, 1.000, [1, 0], 1),
    RunOption(2, 1.000, [1, 1], 1),
    RunOption(2, 1.000, [1, 2], 1),
    RunOption(2, 1.000, [2, 1], 1),
    RunOption(2, 1.000, [2, 2], 1),
    RunOption(2, 14 / (22 - 2sqrt(7)), [0, 1], 1),
    RunOption(2, 14 / (22 - 2sqrt(7)), [1, 0], 1),
    RunOption(2, 14 / (22 - 2sqrt(7)), [1, 1], 1),
    RunOption(2, 14 / (22 - 2sqrt(7)), [1, 2], 1),
    RunOption(2, 14 / (22 - 2sqrt(7)), [2, 1], 1),
    RunOption(2, 14 / (22 - 2sqrt(7)), [2, 2], 1),
    RunOption(2, 56 / 79, [0, 1], 1),
    RunOption(2, 56 / 79, [1, 0], 1),
    RunOption(2, 56 / 79, [1, 1], 1),
    RunOption(2, 56 / 79, [1, 2], 1),
    RunOption(2, 56 / 79, [2, 1], 1),
    RunOption(2, 56 / 79, [2, 2], 1),
    RunOption(3, 0.650, [0, 1], 1),
    RunOption(4, 0.640, [0, 1], 1),
    RunOption(2, 1.000, [0, 1], 2),
    RunOption(2, 1.000, [1, 1], 2),
    RunOption(2, 1.000, [1, 2], 2),
    RunOption(2, 1.000, [2, 1], 2),
    RunOption(2, 1.000, [2, 2], 2),
]

# test range
for runoption in runoptions
    println(
        "p: ",
        runoption.p,
        ", ω: ",
        round(runoption.ω; digits = 3),
        ", v: [",
        runoption.v[1],
        ", ",
        runoption.v[2],
        "], dimension: ",
        runoption.dimension,
    )
    # setup
    mesh = []
    if runoption.dimension == 1
        mesh = Mesh1D(1.0)
    elseif runoption.dimension == 2
        mesh = Mesh2D(1.0, 1.0)
    end
    numbercomponents = 1
    numberfineelements1d = 2
    ctofbasis = TensorH1UniformHProlongationBasis(
        runoption.p + 1,
        numbercomponents,
        runoption.dimension,
        numberfineelements1d,
    )

    # operators
    function diffusionweakform(du::Array{Float64}, w::Array{Float64})
        dv = du * w[1]
        return [dv]
    end
    # -- fine
    basis = TensorH1UniformMacroBasis(
        runoption.p + 1,
        runoption.p + 2,
        numbercomponents,
        runoption.dimension,
        numberfineelements1d,
    )
    inputs = [
        OperatorField(basis, [EvaluationMode.gradient]),
        OperatorField(basis, [EvaluationMode.quadratureweights]),
    ]
    outputs = [OperatorField(basis, [EvaluationMode.gradient])]
    finediffusion = Operator(diffusionweakform, mesh, inputs, outputs)
    # -- coarse
    basis = TensorH1UniformBasis(
        runoption.p + 1,
        runoption.p + 2,
        numbercomponents,
        runoption.dimension,
    )
    inputs = [
        OperatorField(basis, [EvaluationMode.gradient]),
        OperatorField(basis, [EvaluationMode.quadratureweights]),
    ]
    outputs = [OperatorField(basis, [EvaluationMode.gradient])]
    coarsediffusion = Operator(diffusionweakform, mesh, inputs, outputs)

    # -- Jacobi smoother
    jacobi = Jacobi(finediffusion)

    # -- h-multigrid preconditioner
    multigrid = HMultigrid(finediffusion, coarsediffusion, jacobi, [ctofbasis])

    # compute smoothing factor
    # -- setup
    numbersteps = 100
    maxeigenvalue = 0
    θ_min = -π / 2
    θ_max = 3π / 2
    θ_step = 2π / numbersteps
    θ_range = θ_min:θ_step:(θ_max-θ_step)

    θ_maxeigenvalue = -1
    # -- 1D --
    if runoption.dimension == 1
        for i = 1:numbersteps
            θ = [θ_range[i]]
            if abs(θ[1]) > π / 128
                M = computesymbols(multigrid, [runoption.ω], runoption.v, θ)
                eigenvalues = [abs(val) for val in eigvals(M)]
                currentmaxeigenvalue = max(eigenvalues...)
                if (currentmaxeigenvalue > maxeigenvalue)
                    maxeigenvalue = currentmaxeigenvalue
                    θ_maxeigenvalue = θ / π
                end
            end
        end
        # -- 2D --
    elseif runoption.dimension == 2
        for i = 1:numbersteps, j = 1:numbersteps
            θ = [θ_range[i], θ_range[j]]
            if sqrt(abs(θ[1])^2 + abs(θ[2])^2) > π / 128
                M = computesymbols(multigrid, [runoption.ω], runoption.v, θ)
                eigenvalues = [abs(val) for val in eigvals(M)]
                currentmaxeigenvalue = max(eigenvalues...)
                if (currentmaxeigenvalue > maxeigenvalue)
                    maxeigenvalue = currentmaxeigenvalue
                    θ_maxeigenvalue = θ / π
                end
            end
        end
    end
    append!(
        maxeigenvalues,
        DataFrame(
            p = runoption.p,
            ω = runoption.ω,
            θ = θ_maxeigenvalue,
            v = (10 * runoption.v[1] + runoption.v[2]),
            ρ = maxeigenvalue,
        ),
    )
end

CSV.write("two-grid-jacobi-hmultigrid.csv", maxeigenvalues)
