# ------------------------------------------------------------------------------
# No preconditioner
# ------------------------------------------------------------------------------

"""
```julia
IdentityPC()
```

Identity preconditioner to investigate multigrid without smoother

# Returns:

  - identity preconditioner object

# Example:

```jldoctest
# setup
mesh = Mesh2D(1.0, 1.0);
mass = GalleryOperator("mass", 4, 4, mesh);

# preconditioner
identity = IdentityPC(mass);

# verify
println(identity)

# output

identity preconditioner
```
"""
struct IdentityPC <: AbstractPreconditioner
    # data never changes
    operator::Operator

    # inner constructor
    IdentityPC(operator) = new(operator)
end

# printing
# COV_EXCL_START
Base.show(io::IO, preconditioner::IdentityPC) = print(io, "identity preconditioner")
# COV_EXCL_STOP

# COV_EXCL_START
function Base.setproperty!(pc::IdentityPC, f::Symbol, value)
    if f == :operator
        throw(ReadOnlyMemoryError())
    else
        return setfield!(operator, f, value)
    end
end
# COV_EXCL_STOP

# ------------------------------------------------------------------------------
# compute symbols
# ------------------------------------------------------------------------------

"""
```julia
computesymbols(preconditioner, ω, θ)
```

Compute or retrieve the symbol matrix for a identity preconditioned operator

# Arguments:

  - `preconditioner`:  Identity preconditioner to compute symbol matrix for
  - `ω`:               smoothing weighting factor array
  - `θ`:               Fourier mode frequency array (one frequency per dimension)

# Returns:

  - symbol matrix for the identity preconditioner (I)

# Example:

```jldoctest
using LinearAlgebra

# setup
mesh = Mesh2D(1.0, 1.0);
mass = GalleryOperator("mass", 4, 4, mesh);

# preconditioner
identity = IdentityPC(mass);

# compute symbols
A = computesymbols(identity, [], []);

# verify
@assert A ≈ I

# output

```
"""
function computesymbols(preconditioner::IdentityPC, ω::Array, θ::Array)
    # return
    return I
end

# ------------------------------------------------------------------------------
