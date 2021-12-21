# ------------------------------------------------------------------------------
# p-multigrid multilevel example
# ------------------------------------------------------------------------------

using LFAToolkit
using LinearAlgebra

# setup
mesh = Mesh2D(1.0, 1.0)
finep = 4
midp = 2
coarsep = 1
numbercomponents = 1
dimension = 2
ctombasis =
    TensorH1LagrangePProlongationBasis(coarsep + 1, midp + 1, numbercomponents, dimension)
mtofbasis =
    TensorH1LagrangePProlongationBasis(midp + 1, finep + 1, numbercomponents, dimension)

# diffusion operators
finediffusion = GalleryOperator("diffusion", finep + 1, finep + 1, mesh)
middiffusion = GalleryOperator("diffusion", midp + 1, finep + 1, mesh)
coarsediffusion = GalleryOperator("diffusion", coarsep + 1, finep + 1, mesh)

# Chebyshev smoothers
finechebyshev = Chebyshev(finediffusion)
midchebyshev = Chebyshev(middiffusion)

# p-multigrid preconditioner
midmultigrid = PMultigrid(middiffusion, coarsediffusion, midchebyshev, [ctombasis])
multigrid = PMultigrid(finediffusion, midmultigrid, finechebyshev, [mtofbasis])

# compute operator symbols
A = computesymbols(multigrid, [3], [1, 1], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
