# ------------------------------------------------------------------------------
# h-multigrid example
# ------------------------------------------------------------------------------

using LinearAlgebra

# setup
mesh = Mesh2D(1.0, 1.0)
p = 1
numbercomponents = 1
numberfineelements1d = 2
dimension = 2
ctofbasis = TensorH1LagrangeHProlongationBasis(
    p + 1,
    numbercomponents,
    dimension,
    numberfineelements1d,
);

# operators
finediffusion =
    GalleryMacroElementOperator("diffusion", p + 1, p + 2, numberfineelements1d, mesh);
coarsediffusion = GalleryOperator("diffusion", p + 1, p + 2, mesh);

# Chebyshev smoother
chebyshev = Chebyshev(finediffusion)

# h-multigrid preconditioner
multigrid = HMultigrid(finediffusion, coarsediffusion, chebyshev, [ctofbasis])

# compute operator symbols
A = computesymbols(multigrid, [3], [1, 1], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
