# Release Notes

## Current Development

Current development includes:

## v0.6.1

This release fixes static analysis warnings and minor style/formatting inconsistencies.


## v0.6.0

This release includes conformal maps for bases, new operator convenience constructors, and small internal improvements.

Enhancements:

* Added conformal maps `sausage_transformation`, `kosloff_talezer_transformation`, and `hale_trefethen_strip_transformation` for basis quadrature spaces.
* Add advection and SUPG advection operator convenience constructor and examples.
* Switch to `FastGaussQuadrature` package for computing nodes, quadrature points, and quadrature weights.

Bugfixes:

## v0.5.0

This release include some minor fixes and some convenience utilities.

Breaking change:

* Keyword argument indicating quadrature point placement for Basis convenience constructors renamed.

Enhancements:

* Added `computesymbolsoverrange` to facilitate common analysis.

Bugfixes:

* Fix language surrounding quadrature, consistently using Gauss-Legendre and Gauss-Legendre-Lobatto or `gaussquadrature` and `gausslobattoquadrature`.

## v0.4.1

This release includes a small fix for properly handling stretched meshes, such as when `dx != dy`.

## v0.4.0

This release includes improved functionality and bugfixes.

Enhancements:

* Added relaxation parameter for application of BDDC preconditioners

Bugfixes:

* Minor spelling and notation errors corrected
* Newton tolerance for quadrature point computation relaxed
* Identity preconditioner modified for compatibility with multigrid
* Fix computation of change of coordinates for gradient and quadrature weights
* Fix computation of eigenvalue estimates and error iteration for Chebyshev smoother
* Fix injection operator for Dirichlet BDDC

Examples:

* Added linear elasticity example, renumbered Neo-Hookean hyperelasticity example

## v0.3.0

This release includes updated functionality and an interface change.

Breaking change:

* Refactor `GalleryOperator`; now `GalleryVectorOperator` and `GalleryMacroElementOperator` included

Functionality improvement:

* Initial implementations of lumped and Dirichlet BDDC preconditioners added

## v0.2.2

Relax compatibility requirement to Julia 1.3 - 1.6.

## v0.2.1

Minor bugfixes:

* Gauss and Gauss-Lobatto node computation tolerances relaxed
* Typo fixes
* `src` directory reorganized to better support future development

## v0.2

This release includes updated functionality and an improved interface.

Functionality updates include:

* Macro-element bases consisting of multiple micro-elements
* Chebyshev preconditioning analysis
* H-multigrid analysis

Interface improvements include:

* Multi-component basis support simplified

Additional changes include:

* Improved documentation
* Expanded examples, to include Neo-Hookean hyperelasticity
* Rename primary branch to `main`

## v0.1.1

Minor bugfixes.

## v0.1

This release includes initial basic functionality of LFAToolkit.jl.

Functionality includes:

* User defined second order PDEs
* Arbitrary basis order, dimension, and number of components
* Independent mesh scaling in each dimension
* Jacobi preconditioning analysis
* P-multigrid analysis
