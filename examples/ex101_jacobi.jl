# ------------------------------------------------------------------------------
# Jacobi smoother example
# ------------------------------------------------------------------------------

# setup
mesh = Mesh2D(1.0, 1.0)
p = 4

# diffusion operator
diffusion = GalleryOperator("diffusion", p, p, mesh)

# Jacobi smoother
jacobi = Jacobi(diffusion)

# compute operator symbols
A = computesymbols(jacobi, [1.0], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
