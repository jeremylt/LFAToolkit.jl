"""
Finite element basis for function spaces and test spaces
"""
struct Basis
    p::Int
    q::Int
    dimension::Int
    numbercomponents::Int
    istensor::Bool
    nodes::Array{Float64}
    quadraturepoints::Array{Float64}
    quadratureweights::Array{Float64}
    interpolation::Array{Float64,2}
    gradient::Array{Float64,2}
end

"""
Constructor for tensor product basis
"""
function TensorBasis(
    p1d::Int,
    q1d::Int,
    dimension::Int,
    numbercomponents::Int,
    nodes1d::Array{Float64},
    quadraturepoints1d::Array{Float64},
    quadratureweights1d::Array{Float64},
    interpolation1d::Array{Float64,2},
    gradient1d::Array{Float64,2},
)
    Basis(
        p1d,
        q1d,
        dimension,
        numbercomponents,
        true,
        nodes1d,
        quadraturepoints1d,
        quadratureweights1d,
        interpolation1d,
        gradient1d,
    )
end

"""
Constructor for non-tensor basis
"""
function NonTensorBasis(
    p::Int,
    q::Int,
    dimension::Int,
    numbercomponents::Int,
    nodes::Array{Float64},
    quadraturepoints::Array{Float64},
    quadratureweights::Array{Float64},
    interpolation::Array{Float64,2},
    gradient::Array{Float64,2},
)
    Basis(
        p,
        q,
        dimension,
        numbercomponents,
        false,
        nodes,
        quadraturepoints,
        quadratureweights,
        interpolation,
        gradient,
    )
end

"""
    gaussquadtarute()

   Construct a Gauss-Legendre quadrature
"""
function gaussquadrature(q::Int)
    quadraturepoints = zeros(Float64, q)
    quadratureweights = zeros(Float64, q)

    # Bulid qrefid, qweight1d
    for i = 0:floor(Int, q / 2)
        # Guess
        xi = cos(pi * (2 * i + 1.0) / (2 * q))
        # Pn(xi)
        p0 = 1.0
        p1 = xi
        p2 = 0.0
        for j = 2:q
            p2 = ((2 * j - 1.0) * xi * p1 - (j - 1.0) * p0) / j
            p0 = p1
            p1 = p2
        end
        # First Newton Step
        dp2 = (xi * p2 - p0) * q / (xi * xi - 1.0)
        xi = xi - p2 / dp2
        # Newton to convergence
        itr = 0
        while itr < 100 && abs(p2) > 1e-15
            p0 = 1.0
            p1 = xi
            for j = 2:q
                p2 = ((2 * j - 1.0) * xi * p1 - (j - 1.0) * p0) / j
                p0 = p1
                p1 = p2
            end
            dp2 = (xi * p2 - p0) * q / (xi * xi - 1.0)
            xi = xi - p2 / dp2
            itr += 1
        end
        # save xi, wi
        quadraturepoints[i+1] = -xi
        quadraturepoints[q-i] = xi
        wi = 2.0 / ((1.0 - xi * xi) * dp2 * dp2)
        quadratureweights[i+1] = wi
        quadratureweights[q-i] = wi
    end
    return quadraturepoints, quadratureweights
end

"""
    lobattoquadrature()

   Construct a Gauss-Lobatto quadrature
"""
function lobattoquadrature(q::Int, weights::Bool)
    quadraturepoints = zeros(Float64, q)
    quadratureweights = zeros(Float64, q)

    if q < 2
        throw(DomanError())
    end

    # Endpoints
    quadraturepoints[1] = -1.0
    quadraturepoints[q] = 1.0
    if weights
        wi = 2.0 / (q * (q - 1.0))
        quadratureweights[1] = wi
        quadratureweights[q] = wi
    end

    # Bulid qrefid, qweight1d
    for i = 1:floor(Int, (q - 1) / 2)
        # Guess
        xi = cos(pi * i / (q - 1.0))
        # Pn(xi)
        p0 = 1.0
        p1 = xi
        p2 = 0.0
        for j = 2:q-1
            p2 = ((2 * j - 1.0) * xi * p1 - (j - 1.0) * p0) / j
            p0 = p1
            p1 = p2
        end
        # First Newton Step
        dp2 = (xi * p2 - p0) * q / (xi * xi - 1.0)
        d2p2 = (2 * xi * dp2 - q * (q - 1.0) * p2) / (1.0 - xi * xi)
        xi = xi - dp2 / d2p2
        # Newton to convergence
        itr = 0
        while itr < 100 && abs(dp2) > 1e-15
            p0 = 1.0
            p1 = xi
            for j = 2:q-1
                p2 = ((2 * j - 1.0) * xi * p1 - (j - 1.0) * p0) / j
                p0 = p1
                p1 = p2
            end
            dp2 = (xi * p2 - p0) * q / (xi * xi - 1.0)
            d2p2 = (2 * xi * dp2 - q * (q - 1.0) * p2) / (1.0 - xi * xi)
            xi = xi - dp2 / d2p2
            itr += 1
        end
        # save xi, wi
        quadraturepoints[i+1] = -xi
        quadraturepoints[q-i] = xi
        if weights
            wi = 2.0 / (q * (q - 1.0) * p2 * p2)
            quadratureweights[i+1] = wi
            quadratureweights[q-i] = wi
        end
    end
    if weights
        return quadraturepoints, quadratureweights
    else
        return quadraturepoints
    end
end

"""
    TensorH1LagrangeBasis
"""
function TensorH1LagrangeBasis(p1d::Int, q1d::Int, dimension::Int, numbercomponents::Int)
    # Get nodes, quadrature points, and weights
    nodes1d = lobattoquadrature(p1d, false)
    quadraturepoints1d, quadratureweights1d = gaussquadrature(q1d)

    # Build interpolation, gradient matrices
    # Fornberg, 1998
    interpolation1d = zeros(Float64, q1d, p1d)
    gradient1d = zeros(Float64, q1d, p1d)
    for i = 1:q1d
        c1 = 1.0
        c3 = nodes1d[1] - quadraturepoints1d[i]
        interpolation1d[i, 1] = 1.0
        for j = 2:p1d
            c2 = 1.0
            c4 = c3
            c3 = nodes1d[j] - quadraturepoints1d[i]
            for k = 1:j-1
                dx = nodes1d[j] - nodes1d[k]
                c2 *= dx
                if k == j - 1
                    gradient1d[i, j] =
                        c1 * (interpolation1d[i, k] - c4 * gradient1d[i, k]) / c2
                    interpolation1d[i, j] = -c1 * c4 * interpolation1d[i, k] / c2
                end
                gradient1d[i, k] = (c3 * gradient1d[i, k] - interpolation1d[i, k]) / dx
                interpolation1d[i, k] = c3 * interpolation1d[i, k] / dx
            end
            c1 = c2
        end
    end

    # Use Constructor
    return TensorBasis(
        p1d,
        q1d,
        dimension,
        numbercomponents,
        nodes1d,
        quadraturepoints1d,
        quadratureweights1d,
        interpolation1d,
        gradient1d,
    )
end
