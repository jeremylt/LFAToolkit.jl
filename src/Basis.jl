struct Basis
    p::Int
    q::Int
    numbercomponents::Int
    istensor::Bool
    quadraturepoints::Array{Fload64}
    quadratureweights::Array{Fload64}
    interpolation::Array{Fload64}
    gradient::Array{Fload64}
end

function TensorBasis(
    p1d::Int,
    q1d::Int,
    numbercomponents1d::Int,
    quadraturepoints1d::Array{Fload64},
    quadratureweights1d::Array{Fload64},
    interpolation1d::Array{Fload64},
    gradient1d::Array{Fload64},
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
    quadraturepoints::Array{Fload64},
    quadratureweights::Array{Fload64},
    interpolation::Array{Fload64},
    gradient::Array{Fload64},
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
