# ------------------------------------------------------------------------------
# p-multigrid multilevel example
# ------------------------------------------------------------------------------

# setup
mesh = Mesh2D(1.0, 1.0)
finep = 5
midp = 3
coarsep = 2
dimension = 2
ctombasis = TensorH1LagrangePProlongationBasis(coarsep, midp, dimension)
mtofbasis = TensorH1LagrangePProlongationBasis(midp, finep, dimension)

# diffusion operators
finediffusion = GalleryOperator("diffusion", finep, finep, mesh)
middiffusion = GalleryOperator("diffusion", midp, finep, mesh)
coarsediffusion = GalleryOperator("diffusion", coarsep, finep, mesh)

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
