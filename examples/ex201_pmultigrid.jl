# ------------------------------------------------------------------------------
# p-multigrid example
# ------------------------------------------------------------------------------

using LinearAlgebra

# setup
mesh = Mesh2D(1.0, 1.0)
finep = 3
coarsep = 2
numbercomponents = 1
dimension = 2
ctofbasis = TensorH1LagrangeBasis(
    coarsep,
    finep,
    numbercomponents,
    dimension,
    lagrangequadrature = true,
)

# diffusion operators
finediffusion = GalleryOperator("diffusion", finep, finep, mesh)
coarsediffusion = GalleryOperator("diffusion", coarsep, finep, mesh)

# Chebyshev smoother
chebyshev = Chebyshev(finediffusion)

# p-multigrid preconditioner
multigrid = PMultigrid(finediffusion, coarsediffusion, chebyshev, [ctofbasis])

# compute operator symbols
A = computesymbols(multigrid, [3], [1, 1], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
