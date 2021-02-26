# ------------------------------------------------------------------------------
# No preconditioner
# ------------------------------------------------------------------------------

"""
```julia
IdentityPC()
```

Identity preconditioner to investigate multigrid without smoother

# Returns:
- Identity preconditioner object

# Example:
```jldoctest
# preconditioner
identity = IdentityPC();

# verify
println(identity)

# output
identity preconditioner
```
"""
struct IdentityPC <: AbstractPreconditioner
    # no internal data

    # inner constructor
    IdentityPC() = new()
end

# printing
# COV_EXCL_START
Base.show(io::IO, preconditioner::IdentityPC) = print(io, "identity preconditioner")
# COV_EXCL_STOP

# ------------------------------------------------------------------------------
# compute symbols
# ------------------------------------------------------------------------------

"""
```julia
computesymbols(preconditioner, ω, θ)
```

Compute or retrieve the symbol matrix for a Jacobi preconditioned operator

# Arguments:
- `preconditioner`: Identity preconditioner to compute symbol matrix for
- `ω`:              Smoothing weighting factor array
- `θ`:              Fourier mode frequency array (one frequency per dimension)

# Returns:
- Symbol matrix for the identity preconditioner (I)

# Example:
```jldoctest
using LinearAlgebra

# preconditioner
identity = IdentityPC();

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
