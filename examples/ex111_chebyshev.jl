# ------------------------------------------------------------------------------
# Chebyshev smoother example
# ------------------------------------------------------------------------------

# setup
mesh = Mesh2D(1.0, 1.0)
p = 4

# diffusion operator
diffusion = GalleryOperator("diffusion", p, p, mesh)

# Chebyshev smoother
chebyshev = Chebyshev(diffusion)

# compute operator symbols
A = computesymbols(chebyshev, [3], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
