# ---------------------------------------------------------------------------------------------------------------------
# Rectangular meshes
# ---------------------------------------------------------------------------------------------------------------------

"""
Rectangular mesh with independent scaling in each dimesion
"""
abstract type Mesh end

"""
```julia
Mesh1D(dx)
```

One dimensional regular background mesh

# Arguments:
- `dx`: deformation in x dimension

# Returns:
- One dimensional mesh object

# Example:
```jldoctest
# generate 1D mesh
mesh = Mesh1D(1.0);

# verify
@assert mesh.dimension == 1
@assert abs(mesh.dx - 1.0) < 1e-15

# output

```
"""
struct Mesh1D <: Mesh
    # data
    dimension::Int
    dx::Float64

    # inner constructor
    Mesh1D(dx) = (
        # validity checking
        if dx < 1e-14
            error("Mesh scaling must be positive") # COV_EXCL_LINE
        end;

        # constructor
        new(1, dx)
    )
end

"""
```julia
Mesh2D(dx, dy)
```

Two dimensional regular background mesh

# Arguments:
- `dx`: deformation in x dimension
- `dy`: deformation in y dimension

# Returns:
- Two dimensional mesh object

# Example:
```jldoctest
# generate 2D mesh
mesh = Mesh2D(1.0, 0.5);

# verify
@assert mesh.dimension == 2
@assert abs(mesh.dx - 1.0) < 1e-15
@assert abs(mesh.dy - 0.5) < 1e-15

# output

```
"""
struct Mesh2D <: Mesh
    # data
    dimension::Int
    dx::Float64
    dy::Float64

    # inner constructor
    Mesh2D(dx, dy) = (
        # validity checking
        if dx < 1e-14 || dy < 1e-14
            error("Mesh scaling must be positive") # COV_EXCL_LINE
        end;

        # constructor
        new(2, dx, dy)
    )
end

"""
```julia
Mesh3D(dx, dy, dz)
```

Three dimensional regular background mesh

# Arguments:
- `dx`: deformation in x dimension
- `dy`: deformation in y dimension
- `dz`: deformation in z dimension

# Returns:
- Three dimensional mesh object

# Example:
```jldoctest
# generate 3D mesh
mesh = Mesh3D(1.0, 0.5, 0.3);

# verify
@assert mesh.dimension == 3
@assert abs(mesh.dx - 1.0) < 1e-15
@assert abs(mesh.dy - 0.5) < 1e-15
@assert abs(mesh.dz - 0.3) < 1e-15

# output

``` 
"""
struct Mesh3D <: Mesh
    # data
    dimension::Int
    dx::Float64
    dy::Float64
    dz::Float64

    # inner constructor
    Mesh3D(dx, dy, dz) = (
        # validity checking
        if dx < 1e-14 || dy < 1e-14 || dz < 1e-14
            error("Mesh scaling must be positive") # COV_EXCL_LINE
        end;

        # constructor
        new(3, dx, dy, dz)
    )
end

# ---------------------------------------------------------------------------------------------------------------------
