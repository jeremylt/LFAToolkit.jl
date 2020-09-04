# ---------------------------------------------------------------------------------------------------------------------
# Finite element bases
# ---------------------------------------------------------------------------------------------------------------------

"""
Finite element basis for function spaces and test spaces
"""
abstract type Basis end

# ---------------------------------------------------------------------------------------------------------------------
# Basic basis types
# ---------------------------------------------------------------------------------------------------------------------

"""
Tensor product basis
"""
struct TensorBasis <: Basis
    p1d::Int
    q1d::Int
    dimension::Int
    numbercomponents::Int
    nodes1d::Array{Float64}
    quadraturepoints1d::Array{Float64}
    quadratureweights1d::Array{Float64}
    interpolation1d::Array{Float64,2}
    gradient1d::Array{Float64,2}
end

"""
Non-tensor basis
"""
struct NonTensorBasis <: Basis
    p::Int
    q::Int
    dimension::Int
    numbercomponents::Int
    nodes::Array{Float64}
    quadraturepoints::Array{Float64}
    quadratureweights::Array{Float64}
    interpolation::Array{Float64,2}
    gradient::Array{Float64,2}
end

# ---------------------------------------------------------------------------------------------------------------------
# Utility functions for generating polynomial bases
# ---------------------------------------------------------------------------------------------------------------------

"""
    gaussquadrature()

    Construct a Gauss-Legendre quadrature

    ```@meta
    DocTestSetup = quote
      using LFAToolkit
    end
    ```

    ```jldoctest
    quadraturepoints = LFAToolkit.gaussquadrature(5)

    # output
    ([-0.906179845938664, -0.5384693101056831, 1.232595164407831e-32, 0.5384693101056831, 0.906179845938664], [0.2369268850561885, 0.47862867049936647, 0.5688888888888889, 0.47862867049936647, 0.2369268850561885])
    ```

    ```@meta
    DocTestSetup = nothing
    ```
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

    # Return
    return quadraturepoints, quadratureweights
end

"""
    lobattoquadrature()

    Construct a Gauss-Lobatto quadrature

    ```@meta
    DocTestSetup = quote
      using LFAToolkit
    end
    ```

    ```jldoctest
    quadraturepoints = LFAToolkit.lobattoquadrature(5, false)

    # output
    5-element Array{Float64,1}:
    -1.0
    -0.6546536707079771
    -1.530808498934193e-17
     0.6546536707079771
     1.0
    ```
   
    ```jldoctest
    quadraturepoints, quadratureweights = LFAToolkit.lobattoquadrature(5, false)

    # output
    ([-1.0, -0.6546536707079771, -1.530808498934193e-17, 0.6546536707079771, 1.0], [0.1, 0.5444444444444443, 0.7111111111111111, 0.5444444444444443, 0.1])
    ```

    ```@meta
    DocTestSetup = nothing
    ```
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

    # Return
    if weights
        return quadraturepoints, quadratureweights
    else
        return quadraturepoints
    end
end

# ---------------------------------------------------------------------------------------------------------------------
# User utility constructors
# ---------------------------------------------------------------------------------------------------------------------

"""
    TensorH1LagrangeBasis()

    Tensor product basis on Gauss-Lobatto points with Gauss-Legendre quadrature

    ```@meta
    DocTestSetup = quote
        using LFAToolkit
    end
    ```

    ```jldoctest
    basis = TensorH1LagrangeBasis(4, 4, 1, 1)

    # output
    LFAToolkit.TensorBasis(4, 4, 1, 1, [-1.0, -0.4472135954999579, 0.4472135954999579, 1.0], [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526], [0.34785484513745374, 0.6521451548625461, 0.6521451548625461, 0.34785484513745374], [0.6299431661034454 0.472558747113818 -0.14950343104607952 0.04700151782881607; -0.07069479527385582 0.972976186258263 0.13253992624542693 -0.03482131722983419; -0.03482131722983419 0.13253992624542696 0.9729761862582628 -0.07069479527385582; 0.04700151782881607 -0.14950343104607955 0.47255874711381796 0.6299431661034455], [-2.341837415390958 2.787944890537088 -0.6351041115519563 0.18899663640582656; -0.5167021357255352 -0.48795249031352683 1.3379050992756671 -0.3332504732366054; 0.33325047323660545 -1.3379050992756674 0.4879524903135269 0.5167021357255351; -0.1889966364058266 0.6351041115519563 -2.7879448905370876 2.3418374153909585])
    ```

    ```@meta
    DocTestSetup = nothing
    ```
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

# ---------------------------------------------------------------------------------------------------------------------
