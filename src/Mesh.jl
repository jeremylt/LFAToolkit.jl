# ---------------------------------------------------------------------------------------------------------------------
# Rectangular meshes
# ---------------------------------------------------------------------------------------------------------------------

"""
Rectangular mesh with independent scaling in each dimesion
"""
abstract type Mesh end

"""
    Mesh1D()

```jldoctest
mesh = Mesh1D(1.0);

if (mesh.dx - 1.0) > 1e-15
    println("Incorrect x scale")
end

# output

```
"""
struct Mesh1D <: Mesh
    dimension::Int
    dx::Float64
    Mesh1D(dx) = dx > 0 ? new(1, dx) : error("Mesh scaling must be positive")
end

"""
    Mesh2D()

```jldoctest
mesh = Mesh2D(1.0, 0.5);

if (mesh.dx - 1.0) > 1e-15
    println("Incorrect x scale")
end
if (mesh.dy - 0.5) > 1e-15
    println("Incorrect y scale")
end

# output

```
"""
struct Mesh2D <: Mesh
    dimension::Int
    dx::Float64
    dy::Float64
    Mesh2D(dx, dy) =
        dx > 0 && dy > 0 ? new(2, dx, dy) : error("Mesh scaling must be positive")
end

"""
    Mesh2D()

```jldoctest
mesh = Mesh3D(1.0, 0.5, 0.25);

if (mesh.dx - 1.0) > 1e-15
    println("Incorrect x scale")
end
if (mesh.dy - 0.5) > 1e-15
    println("Incorrect y scale")
end
if (mesh.dz - 0.25) > 1e-15
    printlt("Incorrect z scale")
end

# output

``` 
"""
struct Mesh3D <: Mesh
    dimension::Int
    dx::Float64
    dy::Float64
    dz::Float64
    Mesh3D(dx, dy, dz) = dx > 0 && dy > 0 && dz > 0 ? new(3, dx, dy, dz) :
        error("Mesh scaling must be positive")
end

# ---------------------------------------------------------------------------------------------------------------------
