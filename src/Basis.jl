"""
Finite element basis for function spaces and test spaces
"""
struct Basis
    p::Int
    q::Int
    numbercomponents::Int
    istensor::Bool
    quadraturepoints::Array{Float64}
    quadratureweights::Array{Float64}
    interpolation::Array{Float64}
    gradient::Array{Float64}
end

function TensorBasis(
    p1d::Int,
    q1d::Int,
    numbercomponents1d::Int,
    quadraturepoints1d::Array{Float64},
    quadratureweights1d::Array{Float64},
    interpolation1d::Array{Float64},
    gradient1d::Array{Float64},
)
    Basis(
        p1d,
        q1d,
        numbercomponents1d,
        true,
        quadraturepoints1d,
        quadratureweights1d,
        interpolation1d,
        gradient1d,
    )
end

function NonTensorBasis(
    p::Int,
    q::Int,
    numbercomponents::Int,
    quadraturepoints::Array{Float64},
    quadratureweights::Array{Float64},
    interpolation::Array{Float64},
    gradient::Array{Float64},
)
    Basis(
        p,
        q,
        numbercomponents,
        false,
        quadraturepoints,
        quadratureweights,
        interpolation,
        gradient,
    )
end
