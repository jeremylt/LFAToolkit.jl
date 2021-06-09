# ------------------------------------------------------------------------------
# lumped BDDC example
# ------------------------------------------------------------------------------

using LinearAlgebra

# setup
mesh = Mesh2D(1.0, 1.0)
p = 4

# diffusion operator
diffusion = GalleryOperator("diffusion", p, p, mesh)

# lumped BDDC preconditioner
bddc = LumpedBDDC(diffusion)

# compute operator symbols
A = computesymbols(bddc, [0.2], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
