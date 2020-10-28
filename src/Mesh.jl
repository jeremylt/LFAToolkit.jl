# ------------------------------------------------------------------------------
# rectangular meshes
# ------------------------------------------------------------------------------

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
println(mesh)

# output
1d mesh:
    dx: 1.0
```
"""
struct Mesh1D <: Mesh
    # data
    dimension::Int
    dx::Float64
    volume::Float64

    # inner constructor
    Mesh1D(dx::Float64) = (
        # validity checking
        if dx < 1e-14
            error("Mesh scaling must be positive") # COV_EXCL_LINE
        end;

        # constructor
        new(1, dx, dx)
    )
end

# printing
# COV_EXCL_START
Base.show(io::IO, mesh::Mesh1D) = print(
    io,
    "1d mesh:
    dx: ",
    mesh.dx,
)
# COV_EXCL_STOP

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
println(mesh)

# output
2d mesh:
    dx: 1.0
    dy: 0.5
```
"""
struct Mesh2D <: Mesh
    # data
    dimension::Int
    dx::Float64
    dy::Float64
    volume::Float64

    # inner constructor
    Mesh2D(dx::Float64, dy::Float64) = (
        # validity checking
        if dx < 1e-14 || dy < 1e-14
            error("Mesh scaling must be positive") # COV_EXCL_LINE
        end;

        # constructor
        new(2, dx, dy, dx*dy)
    )
end

# printing
# COV_EXCL_START
Base.show(io::IO, mesh::Mesh2D) = print(
    io,
    "2d mesh:
    dx: ",
    mesh.dx,
    "\n    dy: ",
    mesh.dy,
)
# COV_EXCL_STOP

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
println(mesh)

# output
3d mesh:
    dx: 1.0
    dy: 0.5
    dz: 0.3
``` 
"""
struct Mesh3D <: Mesh
    # data
    dimension::Int
    dx::Float64
    dy::Float64
    dz::Float64
    volume::Float64

    # inner constructor
    Mesh3D(dx::Float64, dy::Float64, dz::Float64) = (
        # validity checking
        if dx < 1e-14 || dy < 1e-14 || dz < 1e-14
            error("Mesh scaling must be positive") # COV_EXCL_LINE
        end;

        # constructor
        new(3, dx, dy, dz, dx*dy*dz)
    )
end

# printing
# COV_EXCL_START
Base.show(io::IO, mesh::Mesh3D) = print(
    io,
    "3d mesh:
    dx: ",
    mesh.dx,
    "\n    dy: ",
    mesh.dy,
    "\n    dz: ",
    mesh.dz,
)
# COV_EXCL_STOP

# ------------------------------------------------------------------------------
