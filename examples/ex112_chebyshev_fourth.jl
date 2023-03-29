# ------------------------------------------------------------------------------
# Chebyshev smoother example
# ------------------------------------------------------------------------------

using LFAToolkit
using LinearAlgebra

# setup
mesh = Mesh2D(1.0, 1.0)
p = 3

# diffusion operator
diffusion = GalleryOperator("diffusion", p + 1, p + 1, mesh)

# Chebyshev smoother
chebyshev = Chebyshev(diffusion)
setchebyshevtype(chebyshev, ChebyshevType.fourth)

# compute operator symbols
A = computesymbols(chebyshev, [3], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
