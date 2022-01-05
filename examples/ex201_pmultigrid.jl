# ------------------------------------------------------------------------------
# p-multigrid example
# ------------------------------------------------------------------------------

using LFAToolkit
using LinearAlgebra

# setup
mesh = Mesh2D(1.0, 1.0)
finep = 2
coarsep = 1
numbercomponents = 1
dimension = 2
ctofbasis = TensorH1LagrangeBasis(
    coarsep + 1,
    finep + 1,
    numbercomponents,
    dimension,
    collocatedquadrature = true,
)

# diffusion operators
finediffusion = GalleryOperator("diffusion", finep + 1, finep + 1, mesh)
coarsediffusion = GalleryOperator("diffusion", coarsep + 1, finep + 1, mesh)

# Chebyshev smoother
chebyshev = Chebyshev(finediffusion)

# p-multigrid preconditioner
multigrid = PMultigrid(finediffusion, coarsediffusion, chebyshev, [ctofbasis])

# compute operator symbols
A = computesymbols(multigrid, [3], [1, 1], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
