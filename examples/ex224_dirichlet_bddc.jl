# ------------------------------------------------------------------------------
# lumped BDDC on macro elements example
# ------------------------------------------------------------------------------

using LinearAlgebra

# setup
mesh = Mesh2D(1.0, 1.0)
p = 2
numberelements1d = 4

# operator
diffusion = GalleryMacroElementOperator("diffusion", p, p + 1, numberelements1d, mesh)

# BDDC smoother
bddc = DirichletBDDC(diffusion)

# compute operator symbols
A = computesymbols(bddc, [0.2], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
