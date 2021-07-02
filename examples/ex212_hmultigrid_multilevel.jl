# ------------------------------------------------------------------------------
# h-multigrid multilevel example
# ------------------------------------------------------------------------------

using LinearAlgebra

# setup
mesh = Mesh2D(1.0, 1.0)
p = 1
numberfineelements1d = 4
numbermidelements1d = 2
numbercomponents = 1
dimension = 2
ctombasis = TensorH1LagrangeHProlongationBasis(
    p + 1,
    numbercomponents,
    dimension,
    numbermidelements1d,
);
mtofbasis = TensorH1LagrangeHProlongationMacroBasis(
    p + 1,
    numbercomponents,
    dimension,
    numbermidelements1d,
    numberfineelements1d,
);

# operators
finediffusion =
    GalleryMacroElementOperator("diffusion", p + 1, p + 2, numberfineelements1d, mesh);
middiffusion =
    GalleryMacroElementOperator("diffusion", p + 1, p + 2, numbermidelements1d, mesh);
coarsediffusion = GalleryOperator("diffusion", p + 1, p + 2, mesh);

# Chebyshev smoothers
finechebyshev = Chebyshev(finediffusion)
midchebyshev = Chebyshev(middiffusion)

# h-multigrid preconditioner
midmultigrid = HMultigrid(middiffusion, coarsediffusion, midchebyshev, [ctombasis])
multigrid = HMultigrid(finediffusion, midmultigrid, finechebyshev, [mtofbasis])

# compute operator symbols
A = computesymbols(multigrid, [3], [1, 1], [π, π])
eigenvalues = real(eigvals(A))

# ------------------------------------------------------------------------------
