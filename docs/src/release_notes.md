# Release Notes

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

Additonal changes include:

* Improved documentation
* Expanded examples, to include Neo-Hookean hyperelasticity
* Rename primary branch to `main`

## v0.1.1

Minor bugfixes.

## v0.1

This release includes initial basic functionality of LFAToolkit.jl.

Functionality includes:

* User defined second order PDEs
* Abritrary basis order, dimension, and number of components
* Independent mesh scaling in each dimension
* Jacobi preconditioning analysis
* P-multigrid analysis