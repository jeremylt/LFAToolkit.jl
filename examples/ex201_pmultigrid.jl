# ------------------------------------------------------------------------------
# p-multigrid example
# ------------------------------------------------------------------------------

# setup
mesh = Mesh2D(1.0, 1.0)
finep = 3
coarsep = 2
dimension = 2
ctofbasis = TensorH1LagrangeBasis(coarsep, finep, dimension, lagrangequadrature = true)

# diffusion operators
finediffusion = GalleryOperator("diffusion", finep, finep, mesh)
coarsediffusion = GalleryOperator("diffusion", coarsep, finep, mesh)

# Jacobi smoother
jacobi = Jacobi(finediffusion)

# p-multigrid preconditioner
multigrid = PMultigrid(finediffusion, coarsediffusion, jacobi, [ctofbasis])

# compute operator symbols
A = computesymbols(multigrid, [0.7], [1, 1], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
