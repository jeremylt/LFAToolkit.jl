# ------------------------------------------------------------------------------
# finite element operator gallery
# ------------------------------------------------------------------------------

"""
```julia
GalleryOperator("mass", p1d, q1d, mesh)
```

Convenience constructor for scalar mass operator

# Weak form:
- ``\\int v u``

# Arguments:
- `p1d`:  polynomial order of TensorH1LagrangeBasis
- `q1d`:  number of quadrature points in one dimension for basis
- `mesh`: mesh for operator

# Returns:
- Mass matrix operator of order p on mesh

# Example:
```jldoctest
# mass operator
mesh = Mesh2D(1.0, 1.0);
mass = GalleryOperator("mass", 3, 4, mesh);

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
function massoperator(p1d::Int, q1d::Int, mesh::Mesh)
    # setup
    basis = TensorH1LagrangeBasis(p1d, q1d, 1, mesh.dimension)
    function massweakform(u::Array{Float64}, w::Array{Float64})
        v = u*w[1]
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
GalleryOperator("vectormass", p1d, q1d, mesh)
```

Convenience constructor for vector mass operator in three components

# Weak form:
- ``\\int \\mathbf{v} \\mathbf{u}``

# Arguments:
- `p1d`:  polynomial order of TensorH1LagrangeBasis
- `q1d`:  number of quadrature points in one dimension for basis
- `mesh`: mesh for operator

# Returns:
- Vector mass matrix operator of order p on mesh

# Example:
```jldoctest
# mass operator
mesh = Mesh2D(1.0, 1.0);
mass = GalleryOperator("vectormass", 3, 4, mesh);

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
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    interpolation
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    interpolation
```
"""
function vectormassoperator(p1d::Int, q1d::Int, mesh::Mesh)
    # setup
    basis = TensorH1LagrangeBasis(p1d, q1d, 3, mesh.dimension)
    function massweakform(u::Array{Float64}, w::Array{Float64})
        v = u*w[1]
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
GalleryOperator("diffusion", p1d, q1d, mesh)
```

Convenience constructor for scalar diffusion operator

# Weak form:
- ``\\int \\nabla v \\nabla u``

# Arguments:
- `p1d`:  polynomial order of TensorH1LagrangeBasis
- `q1d`:  number of quadrature points in one dimension for basis
- `mesh`: mesh for operator

# Returns:
- Diffusion operator of order p on mesh

# Example:
```jldoctest
# mass operator
mesh = Mesh2D(1.0, 1.0);
diffusion = GalleryOperator("diffusion", 3, 4, mesh);

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
function diffusionoperator(p1d::Int, q1d::Int, mesh::Mesh)
    # setup
    basis = TensorH1LagrangeBasis(p1d, q1d, 1, mesh.dimension)
    function diffusionweakform(du::Array{Float64}, w::Array{Float64})
        dv = du*w[1]
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
GalleryOperator("vectordiffusion", p1d, q1d, mesh)
```

Convenience constructor for vector diffusion operator in three components

# Weak form:
- ``\\int \\nabla \\mathbf{v} \\nabla \\mathbf{u}``

# Arguments:
- `p1d`:  polynomial order of TensorH1LagrangeBasis
- `q1d`:  number of quadrature points in one dimension for basis
- `mesh`: mesh for operator

# Returns:
- Vector diffusion operator of order p on mesh

# Example:
```jldoctest
# mass operator
mesh = Mesh2D(1.0, 1.0);
diffusion = GalleryOperator("vectordiffusion", 3, 4, mesh);

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
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    gradient
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    quadratureweights

1 output:
operator field:
  tensor product basis:
    numbernodes1d: 3
    numberquadraturepoints1d: 4
    numbercomponents: 3
    dimension: 2
  evaluation mode:
    gradient
```
"""
function vectordiffusionoperator(p1d::Int, q1d::Int, mesh::Mesh)
    # setup
    basis = TensorH1LagrangeBasis(p1d, q1d, 3, mesh.dimension)
    function diffusionweakform(du::Array{Float64}, w::Array{Float64})
        dv = du*w[1]
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

# ------------------------------------------------------------------------------
# operator gallery dictionary
# ------------------------------------------------------------------------------

operatorgallery = Dict(
    "mass" => massoperator,
    "vectormass" => vectormassoperator,
    "diffusion" => diffusionoperator,
    "vectordiffusion" => vectordiffusionoperator,
)

# ------------------------------------------------------------------------------
