# ------------------------------------------------------------------------------
# Chebyshev preconditioner
# ------------------------------------------------------------------------------

"""
```julia
Chebyshev(operator)
```

Chebyshev polynomial preconditioner for finite element operators.
The Chebyshev semi-iterative method is applied to the matrix ``D^{-1} A``, where ``D^{-1}`` is the inverse of the operator diagonal.

# Arguments:

  - `operator::Operator`:  finite element operator to precondition
  - `chebyshevtype::ChebyshevType`: Chebyshev type, first, fourth, or opt. fourth kind

# Returns:

  - Chebyshev preconditioner object

# Example:

```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
mass = GalleryOperator("mass", 4, 4, mesh);

# preconditioner
chebyshev = Chebyshev(mass);

# verify
println(chebyshev)

# output

chebyshev preconditioner:
1st-kind Chebyshev
eigenvalue estimates:
  estimated minimum 0.2500
  estimated maximum 1.3611
estimate scaling:
  λ_min = a * estimated min + b * estimated max
  λ_max = c * estimated min + d * estimated max
  a = 0.0000
  b = 0.1000
  c = 0.0000
  d = 1.0000
```
"""
mutable struct Chebyshev <: AbstractPreconditioner
    # data never changed
    operator::Operator
    eigenvaluebounds::Array{Float64,1}
    type::ChebyshevType.ChebyType

    # data empty until assembled
    eigenvalueestimates::Array{Float64,1}
    operatordiagonalinverse::AbstractArray{Float64}

    # inner constructor
    Chebyshev(
        operator::Operator,
        chebyshevtype::ChebyshevType.ChebyType = ChebyshevType.first,
    ) = new(operator, [0.0, 0.1, 0.0, 1.0], chebyshevtype)
end

# printing
# COV_EXCL_START
function Base.show(io::IO, chebyshev::Chebyshev)
    print(io, "chebyshev preconditioner:")

    # chebyshev type
    if chebyshev.type == ChebyshevType.first
        print(io, "\n1st-kind Chebyshev")
    elseif chebyshev.type == ChebyshevType.fourth
        print(io, "\n4th-kind Chebyshev")
    elseif chebyshev.type == ChebyshevType.opt_fourth
        print(io, "\nOpt. 4th-kind Chebyshev")
    end

    # eigenvalue estimates
    print(io, "\neigenvalue estimates:")
    @printf(io, "\n  estimated minimum %.4f", chebyshev.eigenvalueestimates[1])
    @printf(io, "\n  estimated maximum %.4f", chebyshev.eigenvalueestimates[2])

    # estimate scaling
    print(io, "\nestimate scaling:")
    print(io, "\n  λ_min = a * estimated min + b * estimated max")
    print(io, "\n  λ_max = c * estimated min + d * estimated max")
    @printf(io, "\n  a = %.4f", chebyshev.eigenvaluebounds[1])
    @printf(io, "\n  b = %.4f", chebyshev.eigenvaluebounds[2])
    @printf(io, "\n  c = %.4f", chebyshev.eigenvaluebounds[3])
    @printf(io, "\n  d = %.4f", chebyshev.eigenvaluebounds[4])
end
# COV_EXCL_STOP

# ------------------------------------------------------------------------------
# data for computing symbols
# ------------------------------------------------------------------------------

"""
```julia
getoperatordiagonalinverse(chebyshev)
```

Compute or retrieve the inverse of the symbol matrix diagonal for a Chebyshev preconditioner

# Returns:

  - symbol matrix diagonal inverse for the operator

# Example:

```jldoctest
# setup
mesh = Mesh1D(1.0);
diffusion = GalleryOperator("diffusion", 3, 3, mesh);

# preconditioner
chebyshev = Chebyshev(diffusion);

# verify operator diagonal inverse
@assert chebyshev.operatordiagonalinverse ≈ [6/7 0; 0 3/4]

# output

```
"""
function getoperatordiagonalinverse(chebyshev::Chebyshev)
    # assemble if needed
    if !isdefined(chebyshev, :operatordiagonalinverse)
        # retrieve diagonal and invert
        diagonalinverse = chebyshev.operator.diagonal^-1

        # store
        chebyshev.operatordiagonalinverse = diagonalinverse
    end

    # return
    return getfield(chebyshev, :operatordiagonalinverse)
end

"""
```julia
geteigenvalueestimates(chebyshev)
```

Compute or retrieve the eigenvalue estimates for a Chebyshev preconditioner

# Returns:

  - eigenvalue estimates for the operator

# Example:

```jldoctest
# setup
mesh = Mesh1D(1.0);
diffusion = GalleryOperator("diffusion", 3, 3, mesh);

# preconditioner
chebyshev = Chebyshev(diffusion);

# verify eigenvalue estimates
@assert chebyshev.eigenvalueestimates ≈ [0, 15 / 7]

# output

```
"""
function geteigenvalueestimates(chebyshev::Chebyshev)
    # assemble if needed
    if !isdefined(chebyshev, :eigenvalueestimates)
        dimension = chebyshev.operator.dimension
        λ_min = 1
        λ_max = 0
        θ_step = 2π / 4
        θ_range = -π/2:θ_step:3π/2-θ_step

        # compute eigenvalues
        if dimension == 1
            for θ_x in θ_range
                A = computesymbols(chebyshev.operator, [θ_x])
                eigenvalues = abs.(eigvals(chebyshev.operatordiagonalinverse * A),)
                λ_min = minimum([λ_min, eigenvalues...])
                λ_max = maximum([λ_max, eigenvalues...])
            end
        elseif dimension == 2
            for θ_x in θ_range, θ_y in θ_range
                A = computesymbols(chebyshev.operator, [θ_x, θ_y])
                eigenvalues = abs.(eigvals(chebyshev.operatordiagonalinverse * A),)
                λ_min = minimum([λ_min, eigenvalues...])
                λ_max = maximum([λ_max, eigenvalues...])
            end
        elseif dimension == 3
            for θ_x in θ_range, θ_y in θ_range, θ_z in θ_range
                A = computesymbols(chebyshev.operator, [θ_x, θ_y, θ_z])
                eigenvalues = abs.(eigvals(chebyshev.operatordiagonalinverse * A),)
                λ_min = minimum([λ_min, eigenvalues...])
                λ_max = maximum([λ_max, eigenvalues...])
            end
        end

        # store
        chebyshev.eigenvalueestimates = [λ_min, λ_max]
    end

    # return
    return getfield(chebyshev, :eigenvalueestimates)
end

"""
```julia
seteigenvalueestimatescaling(chebyshev, eigenvaluescaling)
```

Set the scaling of the eigenvalue estimates for a Chebyshev preconditioner

# Arguments:

  - `chebyshev::Chebyshev`:                preconditioner to set eigenvalue estimate scaling
  - `eigenvaluescaling::Array{Float64,1}`:  array of 4 scaling factors to use when setting ``\\lambda_{\\text{min}}`` and ``\\lambda_{\\text{max}}`` based on eigenvalue estimates

``\\lambda_{\\text{min}}`` = a * estimated min + b * estimated max

``\\lambda_{\\text{max}}`` = c * estimated min + d * estimated max

# Example:

```jldoctest
# setup
mesh = Mesh1D(1.0);
diffusion = GalleryOperator("diffusion", 3, 3, mesh);

# preconditioner
chebyshev = Chebyshev(diffusion)

# set eigenvalue estimate scaling
# PETSc default is to use 1.1, 0.1 of max eigenvalue estimate
#   https://www.mcs.anl.gov/petsc/petsc-3.3/docs/manualpages/KSP/KSPChebyshevSetEstimateEigenvalues.html
chebyshev.eigenvaluescaling = [0.0, 0.1, 0.0, 1.1];
println(chebyshev)

# output

chebyshev preconditioner:
1st-kind Chebyshev
eigenvalue estimates:
  estimated minimum 0.0000
  estimated maximum 2.1429
estimate scaling:
  λ_min = a * estimated min + b * estimated max
  λ_max = c * estimated min + d * estimated max
  a = 0.0000
  b = 0.1000
  c = 0.0000
  d = 1.1000
```
"""
function seteigenvalueestimatescaling(
    chebyshev::Chebyshev,
    eigenvaluescaling::Array{Float64,1},
)
    if length(eigenvaluescaling) != 4
        throw(error("exactly four transformation arguments are required")) # COV_EXCL_LINE
    end

    chebyshev.eigenvaluebounds[1] = eigenvaluescaling[1]
    chebyshev.eigenvaluebounds[2] = eigenvaluescaling[2]
    chebyshev.eigenvaluebounds[3] = eigenvaluescaling[3]
    chebyshev.eigenvaluebounds[4] = eigenvaluescaling[4]
    eigenvaluescaling
end

"""
```julia
getbetas(k)
```

Compute beta coefficients associated with fourth/optimal fourth-kind Chebyshev polynomials

# Returns:

  - beta coefficients associated with fourth/optimal fourth-kind Chebyshev polynomials
"""
function getbetas(k::Real)
    if k == 1
        return [1.12500000000000]
    elseif k == 2
        return [
            1.02387287570313
            1.26408905371085
        ]
    elseif k == 3
        return [
            1.00842544782028
            1.08867839208730
            1.33753125909618
        ]
    elseif k == 4
        return [
            1.00391310427285
            1.04035811188593
            1.14863498546254
            1.38268869241000
        ]
    elseif k == 5
        return [
            1.00212930146164
            1.02173711549260
            1.07872433192603
            1.19810065292663
            1.41322542791682
        ]
    elseif k == 6
        return [
            1.00128517255940
            1.01304293035233
            1.04678215124113
            1.11616489419675
            1.23829020218444
            1.43524297106744
        ]
    elseif k == 7
        return [
            1.00083464397912
            1.00843949430122
            1.03008707768713
            1.07408384092003
            1.15036186707366
            1.27116474046139
            1.45186658649364
        ]
    elseif k == 8
        return [
            1.00057246631197
            1.00577427662415
            1.02050187922941
            1.05019803444565
            1.10115572984941
            1.18086042806856
            1.29838585382576
            1.46486073151099
        ]
    elseif k == 9
        return [
            1.00040960072832
            1.00412439506106
            1.01460212148266
            1.03561113626671
            1.07139972529194
            1.12688273710962
            1.20785219140729
            1.32121930716746
            1.47529642820699
        ]
    elseif k == 10
        return [
            1.00030312229652
            1.00304840660796
            1.01077022715387
            1.02619011597640
            1.05231724933755
            1.09255743207549
            1.15083376663972
            1.23172250870894
            1.34060802024460
            1.48386124407011
        ]
    elseif k == 11
        return [
            1.00023058595209
            1.00231675024028
            1.00817245396304
            1.01982986566342
            1.03950210235324
            1.06965042700541
            1.11305754295742
            1.17290876275564
            1.25288300576792
            1.35725579919519
            1.49101672564139
        ]
    elseif k == 12
        return [
            1.00017947200828
            1.00180189139619
            1.00634861907307
            1.01537864566306
            1.03056942830760
            1.05376019693943
            1.08699862592072
            1.13259183097913
            1.19316273358172
            1.27171293675110
            1.37169337969799
            1.49708418575562
        ]
    elseif k == 13
        return [
            1.00014241921559
            1.00142906932629
            1.00503028986298
            1.01216910518495
            1.02414874342792
            1.04238158880820
            1.06842008128700
            1.10399010936759
            1.15102748242645
            1.21171811910125
            1.28854264865128
            1.38432619380991
            1.50229418757368
        ]
    elseif k == 14
        return [
            1.00011490538261
            1.00115246376914
            1.00405357333264
            1.00979590573153
            1.01941300472994
            1.03401425035436
            1.05480599606629
            1.08311420301813
            1.12040891660892
            1.16833095655446
            1.22872122288238
            1.30365305707817
            1.39546814053678
            1.50681646209583
        ]
    elseif k == 15
        return [
            1.00009404750752
            1.00094291696343
            1.00331449056444
            1.00800294833816
            1.01584236259140
            1.02772083317705
            1.04459535422831
            1.06750761206125
            1.09760092545889
            1.13613855366157
            1.18452361426236
            1.24432087304475
            1.31728069083392
            1.40536543893560
            1.51077872501845
        ]
    elseif k == 16
        return [
            1.00007794828179
            1.00078126847253
            1.00274487974401
            1.00662291017015
            1.01309858836971
            1.02289448329337
            1.03678321409983
            1.05559875719896
            1.08024848405560
            1.11172607131497
            1.15112543431072
            1.19965584614973
            1.25865841744946
            1.32962412656664
            1.41421360695576
            1.51427891730346
        ]
    end

    throw(ArgumentError()) # COV_EXCL_LINE
end

# ------------------------------------------------------------------------------
# get/set property
# ------------------------------------------------------------------------------

function Base.getproperty(preconditioner::Chebyshev, f::Symbol)
    if f == :operatordiagonalinverse
        return getoperatordiagonalinverse(preconditioner)
    elseif f == :eigenvalueestimates
        return geteigenvalueestimates(preconditioner)
    else
        return getfield(preconditioner, f)
    end
end

function Base.setproperty!(preconditioner::Chebyshev, f::Symbol, value)
    if f == :operator
        throw(ReadOnlyMemoryError()) # COV_EXCL_LINE
    elseif f == :eigenvaluebounds
        throw(ReadOnlyMemoryError()) # COV_EXCL_LINE
    elseif f == :eigenvaluescaling
        return seteigenvalueestimatescaling(preconditioner, value)
    else
        return setfield!(preconditioner, f, value)
    end
end

# ------------------------------------------------------------------------------
# compute symbols
# ------------------------------------------------------------------------------

"""
```julia
computesymbols(chebyshev, ω, θ)
```

Compute or retrieve the symbol matrix for a Chebyshev preconditioned operator

# Arguments:

  - `chebyshev::Chebyshev`:  Chebyshev preconditioner to compute symbol matrix for
  - `ω::Array{Real}`:        smoothing parameter array [degree], [degree, ``\\lambda_{\\text{max}}``], or [degree, ``\\lambda_{\\text{min}}``, ``\\lambda_{\\text{max}}``]
  - `θ::Array{Real}`:        Fourier mode frequency array (one frequency per dimension)

# Returns:

  - symbol matrix for the Chebyshev preconditioned operator

# Example:

```jldoctest
using LinearAlgebra

for dimension = 1:3
    # setup
    mesh = []
    if dimension == 1
        mesh = Mesh1D(1.0)
    elseif dimension == 2
        mesh = Mesh2D(1.0, 1.0)
    elseif dimension == 3
        mesh = Mesh3D(1.0, 1.0, 1.0)
    end
    diffusion = GalleryOperator("diffusion", 3, 3, mesh)

    # preconditioner
    chebyshev = Chebyshev(diffusion)

    # compute symbols
    A = computesymbols(chebyshev, [1], π * ones(dimension))

    # verify
    eigenvalues = real(eigvals(A))
    if dimension == 1
        @assert minimum(eigenvalues) ≈ 0.15151515151515105
        @assert maximum(eigenvalues) ≈ 0.27272727272727226
    elseif dimension == 2
        @assert minimum(eigenvalues) ≈ -0.25495098334134725
        @assert maximum(eigenvalues) ≈ -0.17128758445192374
    elseif dimension == 3
        @assert minimum(eigenvalues) ≈ -0.8181818181818181
        @assert maximum(eigenvalues) ≈ -0.357575757575757
    end
end

# output

```
"""
function computesymbols(chebyshev::Chebyshev, ω::Array{<:Real}, θ::Array{<:Real})
    # validate number of parameters
    if length(ω) < 1
        throw(error("at least one parameter required for Chebyshev smoothing")) # COV_EXCL_LINE
    elseif length(ω) > 3
        throw(error("no more than three parameters allowed for Chebyshev smoothing")) # COV_EXCL_LINE
    end
    if (ω[1] % 1) > 1E-14 || ω[1] < 1
        throw(error("first parameter must be degree of Chebyshev smoother")) # COV_EXCL_LINE
    end

    # get operator symbol
    A = computesymbols(chebyshev.operator, θ)

    # set eigenvalue estimates
    λ_min = 0
    λ_max = 0
    if length(ω) == 1
        λ_min = chebyshev.eigenvalueestimates[1]
        λ_max = chebyshev.eigenvalueestimates[2]
    elseif length(ω) == 2
        λ_max = ω[2]
    else
        λ_min = ω[2]
        λ_max = ω[3]
    end
    lower = λ_min * chebyshev.eigenvaluebounds[1] + λ_max * chebyshev.eigenvaluebounds[2]
    upper = λ_min * chebyshev.eigenvaluebounds[3] + λ_max * chebyshev.eigenvaluebounds[4]

    # compute Chebyshev smoother of given degree
    D_inv = chebyshev.operatordiagonalinverse
    D_inv_A = D_inv * A
    k = ω[1] # degree of Chebyshev smoother
    if chebyshev.type == ChebyshevType.first
        α = (upper + lower) / 2
        c = (upper - lower) / 2
        β_0 = -c^2 / (2 * α)
        γ_1 = -(α + β_0)
        E_0 = I
        E_1 = I - (1 / α) * D_inv_A * E_0
        E_n = I
        for _ = 2:k
            E_n = (D_inv_A * E_1 - α * E_1 - β_0 * E_0) / γ_1
            β_0 = (c / 2)^2 / γ_1
            γ_1 = -(α + β_0)
            E_0 = E_1
            E_1 = E_n
        end
        # return
        return E_1
    elseif chebyshev.type == ChebyshevType.fourth
        argument = I - 2 * D_inv_A / upper
        W_0 = I
        W_1 = 2 * argument + I
        W_n = I
        for _ = 2:k
            W_n = 2 * argument * W_1 - W_0
            W_0 = W_1
            W_1 = W_n
        end
        # return
        return W_1 / (2 * k + 1)
    elseif chebyshev.type == ChebyshevType.opt_fourth
        argument = I - 2 * D_inv_A / upper
        W_0 = I
        W_1 = 2 * argument + I
        W_n = I
        betas = getbetas(k)
        append!(betas, 0)
        P_k = (1 - betas[1]) * W_0 + (betas[1] - betas[2]) / 3 * W_1
        for n = 2:k
            W_n = 2 * argument * W_1 - W_0
            P_k = P_k + (betas[n] - betas[n+1]) / (2 * n + 1) * W_n
            W_0 = W_1
            W_1 = W_n
        end
        # return
        return P_k
    end
end

# ------------------------------------------------------------------------------
