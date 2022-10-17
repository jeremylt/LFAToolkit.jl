# ------------------------------------------------------------------------------
# finite element operator gallery
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# user utility constructors
# ------------------------------------------------------------------------------

"""
```julia
GalleryOperator(
    name,
    numbernodes1d,
    numberquadraturepoints1d,
    mesh;
    collocatedquadrature = false,
    mapping = nothing,
    parameters = nothing
)
```

Finite element operator from a gallery of options

# Arguments:
- `name`:                      string containing name of operator
- `numbernodes1d`:             polynomial order of TensorH1LagrangeBasis
- `numberquadraturepoints1d`:  number of quadrature points in one dimension for basis
- `mesh`:                      mesh for operator

# Keyword Arguments:
- `collocatedquadrature = false`:  Gauss-Legendre or Gauss-Legendre-Lobatto quadrature points,
                                       default: false, Gauss-Legendre-Lobatto
- `mapping = nothing`:             quadrature point mapping - sausage, Kosloff-Talezer,
                                       or Hale-Trefethen strip transformation
                                       default: nothing, no transformation
- `parameters = nothing`:          named tuple of model parameters
                                       default: nothing, default parameters

# Returns:
- Finite element operator object

# Mass matrix example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
mass = GalleryOperator("mass", 4, 4, mesh);

# verify
println(mass)

# output
finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    interpolation
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    interpolation
```

# Diffusion operator example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
diffusion = GalleryOperator("diffusion", 4, 4, mesh);

# verify
println(diffusion)

# output
finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    gradient
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    gradient
```

# Advection operator example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
advection = GalleryOperator("advection", 4, 4, mesh);

# verify
println(advection)

# output

finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    interpolation
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    gradient
```

# SUPG mass matrix example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
parameters = (wind = [1., 1.], τ = 1.0);
mapping = nothing;
collocate = false;
supgmass = GalleryOperator("supgmass", 4, 4, mesh, parameters = parameters, collocatedquadrature = collocate, mapping = mapping);

# verify
println(supgmass)

# output

finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    interpolation
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation modes:
    interpolation
    gradient
```

# SUPG advection operator example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
parameters = nothing;
mapping = nothing;
collocate = false;
supgadvection = GalleryOperator("supgadvection", 4, 4, mesh, parameters = parameters, collocatedquadrature = collocate, mapping = mapping);

# verify
println(supgadvection)

# output

finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation modes:
    interpolation
    gradient
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    gradient
```

"""
function GalleryOperator(
    name::String,
    numbernodes1d::Int,
    numberquadraturepoints1d::Int,
    mesh::Mesh;
    collocatedquadrature::Bool = false,
    mapping::Union{Tuple{Function,Function},Nothing} = nothing,
    parameters::Union{NamedTuple,Nothing} = nothing,
)
    if haskey(operatorgallery, name)
        basis = TensorH1LagrangeBasis(
            numbernodes1d,
            numberquadraturepoints1d,
            1,
            mesh.dimension,
            collocatedquadrature = collocatedquadrature,
            mapping = mapping,
        )
        if isnothing(parameters)
            return operatorgallery[name](basis, mesh)
        else
            return operatorgallery[name](basis, mesh; parameters = parameters)
        end
    else
        throw(ArgumentError("operator name not found")) # COV_EXCL_LINE
    end
end

"""
```julia
GalleryVectorOperator(name, numbernodes1d, numberquadraturepoints1d, numberelements1d, mesh)
```

Finite element operator from a gallery of options

# Arguments:
- `name`:                      string containing name of operator
- `numbernodes1d`:             polynomial order of TensorH1LagrangeBasis
- `numberquadraturepoints1d`:  number of quadrature points in one dimension for basis
- `numbercomponents`:          number of components
- `mesh`:                      mesh for operator

# Returns:
- Finite element operator object

# Mass matrix example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
mass = GalleryVectorOperator("mass", 4, 4, 3, mesh);

# verify
println(mass)

# output
finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    interpolation
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    interpolation
```

# Diffusion operator example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
diffusion = GalleryVectorOperator("diffusion", 4, 4, 3, mesh);

# verify
println(diffusion)

# output
finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    gradient
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    gradient
```

# Advection operator example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
advection = GalleryVectorOperator("advection", 4, 4, 3, mesh);

# verify
println(advection)

# output

finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    interpolation
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    gradient
```
"""
function GalleryVectorOperator(
    name::String,
    numbernodes1d::Int,
    numberquadraturepoints1d::Int,
    numbercomponents::Int,
    mesh::Mesh,
)
    if haskey(operatorgallery, name)
        basis = TensorH1LagrangeBasis(
            numbernodes1d,
            numberquadraturepoints1d,
            numbercomponents,
            mesh.dimension,
        )
        return operatorgallery[name](basis, mesh)
    else
        throw(ArgumentError("operator name not found")) # COV_EXCL_LINE
    end
end

"""
```julia
GalleryMacroElementOperator(
    name,
    numbernodes1d,
    numberquadraturepoints1d,
    numberelements1d,
    mesh;
    parameters = nothing,
)
```

Finite element operator from a gallery of options

# Arguments:
- `name`:                      string containing name of operator
- `numbernodes1d`:             polynomial order of TensorH1LagrangeBasis
- `numberquadraturepoints1d`:  number of quadrature points in one dimension for basis
- `numberelements1d`:          number of elements in macro-element
- `mesh`:                      mesh for operator

# Keyword Arguments:
- `parameters` = nothing:      named tuple of model parameters
                                   default: nothing, default parameters

# Returns:
- Finite element operator object

# Mass matrix example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
mass = GalleryMacroElementOperator("mass", 4, 4, 2, mesh);

# verify
println(mass)

# output
finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
  macro-element tensor product basis:
    numbernodes1d: 7
    numberquadraturepoints1d: 8
    numbercomponents: 1
    numberelements1d: 2
    dimension: 2
  evaluation mode:
    interpolation
operator field:
  macro-element tensor product basis:
    numbernodes1d: 7
    numberquadraturepoints1d: 8
    numbercomponents: 1
    numberelements1d: 2
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  macro-element tensor product basis:
    numbernodes1d: 7
    numberquadraturepoints1d: 8
    numbercomponents: 1
    numberelements1d: 2
    dimension: 2
  evaluation mode:
    interpolation
```

# Diffusion operator example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
diffusion = GalleryMacroElementOperator("diffusion", 4, 4, 2, mesh);

# verify
println(diffusion)

# output
finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
  macro-element tensor product basis:
    numbernodes1d: 7
    numberquadraturepoints1d: 8
    numbercomponents: 1
    numberelements1d: 2
    dimension: 2
  evaluation mode:
    gradient
operator field:
  macro-element tensor product basis:
    numbernodes1d: 7
    numberquadraturepoints1d: 8
    numbercomponents: 1
    numberelements1d: 2
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  macro-element tensor product basis:
    numbernodes1d: 7
    numberquadraturepoints1d: 8
    numbercomponents: 1
    numberelements1d: 2
    dimension: 2
  evaluation mode:
    gradient
```

# Advection operator example:
```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
advection = GalleryVectorOperator("advection", 4, 4, 3, mesh);

# verify
println(advection)

# output

finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    interpolation
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 4
    numberquadraturepoints1d: 4
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    gradient
```
"""
function GalleryMacroElementOperator(
    name::String,
    numbernodes1d::Int,
    numberquadraturepoints1d::Int,
    numberelements1d::Int,
    mesh::Mesh;
    mapping::Union{Tuple{Function,Function},Nothing} = nothing,
    parameters::Union{NamedTuple,Nothing} = nothing,
)
    if haskey(operatorgallery, name)
        basis = TensorH1LagrangeMacroBasis(
            numbernodes1d,
            numberquadraturepoints1d,
            1,
            mesh.dimension,
            numberelements1d,
        )
        if isnothing(parameters)
            return operatorgallery[name](basis, mesh)
        else
            return operatorgallery[name](basis, mesh; parameters = parameters)
        end
    else
        throw(ArgumentError("operator name not found")) # COV_EXCL_LINE
    end
end

# ------------------------------------------------------------------------------
# Operator Gallery
# ------------------------------------------------------------------------------

"""
```julia
massoperator(basis, mesh; parameters)
```

Convenience constructor for mass operator

# Weak form:
- ``\\int v u``

# Arguments:
- `basis`:  basis for all operator fields to use
- `mesh`:   mesh for operator

# Keyword Arguments:
- `parameters = nothing`:  named tuple of model parameters
                               default: nothing, no parameters

# Returns:
- Mass matrix operator with basis on mesh

# Example:
```jldoctest
# mass operator
mesh = Mesh2D(1.0, 1.0);
basis = TensorH1LagrangeBasis(3, 4, 1, mesh.dimension);
mass = LFAToolkit.massoperator(basis, mesh);

# verify
println(mass)

# output
finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    interpolation
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    interpolation
```
"""
function massoperator(basis::AbstractBasis, mesh::Mesh; parameters = nothing)
    # setup
    function massweakform(u::Array{Float64}, w::Array{Float64})
        v = u * w[1]
        return [v]
    end

    # fields
    inputs = [
        OperatorField(basis, [EvaluationMode.interpolation]),
        OperatorField(basis, [EvaluationMode.quadratureweights]),
    ]
    outputs = [OperatorField(basis, [EvaluationMode.interpolation])]

    # operator
    mass = Operator(massweakform, mesh, inputs, outputs)
    return mass
end

"""
```julia
diffusionoperator(basis, mesh; parameters)
```

Convenience constructor for diffusion operator

# Weak form:
- ``\\int \\nabla v \\nabla u``

# Arguments:
- `basis`:  basis for all operator fields to use
- `mesh`:   mesh for operator

# Keyword Arguments:
- `parameters = nothing`:  named tuple of model parameters
                               default: nothing, no parameters

# Returns:
- Diffusion operator with basis on mesh

# Example:
```jldoctest
# diffusion operator
mesh = Mesh2D(1.0, 1.0);
basis = TensorH1LagrangeBasis(3, 4, 1, mesh.dimension);
diffusion = LFAToolkit.diffusionoperator(basis, mesh);

# verify
println(diffusion)

# output

finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    gradient
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    gradient
```
"""
function diffusionoperator(basis::AbstractBasis, mesh::Mesh; parameters = nothing)
    # setup
    function diffusionweakform(du::Array{Float64}, w::Array{Float64})
        dv = du * w[1]
        return [dv]
    end

    # fields
    inputs = [
        OperatorField(basis, [EvaluationMode.gradient]),
        OperatorField(basis, [EvaluationMode.quadratureweights]),
    ]
    outputs = [OperatorField(basis, [EvaluationMode.gradient])]

    # operator
    diffusion = Operator(diffusionweakform, mesh, inputs, outputs)
    return diffusion
end

"""
```julia
advectionoperator(basis, mesh; parameters)
```
Convenience constructor for advection operator

# Weak form:
- ``\\int \\nabla v u``

# Arguments:
- `basis`:  basis for all operator fields to use
- `mesh`:   mesh for operator

# Keyword Arguments:
- `parameters = ([wind = [1., 1.],)`:  named tuple of model parameters, defines wind speed
                                           default: wind speed = [1.0, 1.0]

# Returns:
- Advection operator with basis on mesh

# Example:
```jldoctest
# advection operator
mesh = Mesh2D(1.0, 1.0);
mapping = hale_trefethen_strip_transformation(1.4);
basis = TensorH1LagrangeBasis(3, 4, 1, mesh.dimension, collocatedquadrature = false, mapping = mapping);
parameters = (wind = [1., 1.],);
advection = LFAToolkit.advectionoperator(basis, mesh; parameters = parameters);

# verify
println(advection)

# output

finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    interpolation
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    gradient
```
"""
function advectionoperator(
    basis::AbstractBasis,
    mesh::Mesh;
    parameters::NamedTuple = (wind = [1.0, 1.0],),
)
    # check parameters
    if !haskey(parameters, :wind)
        parameters.wind = [1.0, 1.0] # COV_EXCL_LINE
    end
    if basis.numbercomponents != 1
        throw(DomainError(basis.numbercomponents, "Advection gallery operator only accepts single component bases")) # COV_EXCL_LINE
    end

    # set up
    function advectionweakform(u::Array{Float64}, w::Array{Float64})
        dv = parameters.wind * u * w[1]
        return [dv]
    end

    # fields
    inputs = [
        OperatorField(basis, [EvaluationMode.interpolation]),
        OperatorField(basis, [EvaluationMode.quadratureweights]),
    ]
    outputs = [OperatorField(basis, [EvaluationMode.gradient])]

    # operator
    advection = Operator(advectionweakform, mesh, inputs, outputs)
    return advection
end

"""
```julia
supgmassoperator(basis, mesh; parameters)
```
Convenience constructor for SUPG mass matrix operator

# Weak form: Left hand side
- ``\\int v u_t + wind τ u_t \\nabla v``

# Arguments:
- `basis`:  basis for all operator fields to use
- `mesh`:   mesh for operator

# Keyword Arguments:
- `parameters = ([wind = [1., 1.], τ = 1.0)`:  named tuple of model parameters, defines wind speed and SUPG scaling
                                                   default: wind speed = [1.0, 1.0], τ = 1.0

# Returns:
- SUPG mass matrix operator with basis on mesh

# Example:
```jldoctest
# supg mass matrix operator
mesh = Mesh2D(1.0, 1.0);
mapping = nothing;
basis = TensorH1LagrangeBasis(3, 4, 1, mesh.dimension, collocatedquadrature = false, mapping = mapping);
parameters = (wind = [1., 1.], τ = 1.0);
supgmass = LFAToolkit.supgmassoperator(basis, mesh; parameters = parameters);

# verify
println(supgmass)

# output

finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    interpolation
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation modes:
    interpolation
    gradient
```
"""
function supgmassoperator(
    basis::AbstractBasis,
    mesh::Mesh;
    parameters::NamedTuple = (wind = [1.0, 1.0], τ = 1.0),
)
    # check parameters
    if !haskey(parameters, :wind)
        parameters.wind = [1.0, 1.0] # COV_EXCL_LINE
    end
    if !haskey(parameters, :τ)
        parameters.τ = 1.0 # COV_EXCL_LINE
    end

    # set up
    function supgmassweakform(udot::Array{Float64}, w::Array{Float64})
        v = udot * w[1]
        dv = parameters.wind * parameters.τ * udot * w[1]
        return [v; dv]
    end

    # fields
    inputs = [
        OperatorField(basis, [EvaluationMode.interpolation]),
        OperatorField(basis, [EvaluationMode.quadratureweights]),
    ]
    outputs =
        [OperatorField(basis, [EvaluationMode.interpolation, EvaluationMode.gradient])]

    #operator
    supgmass = Operator(supgmassweakform, mesh, inputs, outputs)
    return supgmass
end

"""
```julia
supgadvectionoperator(basis, mesh; parameters)
```
Convenience constructor for SUPG advection operator

# Weak form: Right hand side
- ``\\int wind \\nabla v u - wind wind τ \\nabla v \\nabla u``

# Arguments:
- `basis`:  basis for all operator fields to use
- `mesh`:   mesh for operator

# Keyword Arguments:
- `parameters = ([wind = [1., 1.], τ = 1.0)`:  named tuple of model parameters, defines wind speed and SUPG scaling
                                                   default: wind speed = [1.0, 1.0], τ = 1.0

# Returns:
- SUPG advection operator with basis on mesh

# Example:
```jldoctest
# supg advection operator
mesh = Mesh2D(1.0, 1.0);
mapping = nothing;
basis = TensorH1LagrangeBasis(3, 4, 1, mesh.dimension, collocatedquadrature = false, mapping = mapping);
parameters = (wind = [1., 1.], τ = 1.0);
supgadvection = LFAToolkit.supgadvectionoperator(basis, mesh; parameters = parameters);

# verify
println(supgadvection)

# output

finite element operator:
2d mesh:
    dx: 1.0
    dy: 1.0

2 inputs:
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation modes:
    interpolation
    gradient
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 1
    dimension: 2
  evaluation mode:
    gradient
```
"""
function supgadvectionoperator(
    basis::AbstractBasis,
    mesh::Mesh;
    parameters::NamedTuple = (wind = [1.0, 1.0], τ = 1.0),
)
    # check parameters
    if !haskey(parameters, :wind)
        parameters.wind = [1.0, 1.0] # COV_EXCL_LINE
    end
    if !haskey(parameters, :τ)
        parameters.τ = 1.0 # COV_EXCL_LINE
    end

    # set up
    function supgadvectionweakform(U::Matrix{Float64}, w::Array{Float64})
        u = U[1, :]
        du = U[2, :]
        dv =
            (
                hcat(parameters.wind * u', [0; 0]) -
                parameters.τ * parameters.wind * parameters.wind' .* du
            ) .* w[1]
        return [dv]
    end

    # fields
    inputs = [
        OperatorField(basis, [EvaluationMode.interpolation, EvaluationMode.gradient]),
        OperatorField(basis, [EvaluationMode.quadratureweights]),
    ]
    outputs = [OperatorField(basis, [EvaluationMode.gradient])]

    # operator
    supgadvection = Operator(supgadvectionweakform, mesh, inputs, outputs)
    return supgadvection
end

# ------------------------------------------------------------------------------
# operator gallery dictionary
# ------------------------------------------------------------------------------

operatorgallery = Dict(
    "mass" => massoperator,
    "diffusion" => diffusionoperator,
    "advection" => advectionoperator,
    "supgmass" => supgmassoperator,
    "supgadvection" => supgadvectionoperator,
)

# ------------------------------------------------------------------------------
