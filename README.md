# LFAToolkit

[![CI Status](https://github.com/jeremylt/LFAToolkit.jl/workflows/ci/badge.svg)](https://github.com/jeremylt/LFAToolkit.jl/actions)
[![CodeCov](https://codecov.io/gh/jeremylt/LFAToolkit.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jeremylt/LFAToolkit.jl)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Documentation](https://img.shields.io/badge/docs-dev-blue)](https://jeremylt.github.io/LFAToolkit.jl/dev/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jeremylt/LFAToolkit.jl/master?filepath=examples%2Fjupyter)

Local Fourier Analysis for arbitrary order finite element type operators

## Introduction

Local Fourier Analysis is a tool commonly used in the analysis of multigrid and multilevel algorthms for solving partial differential equations via finite element or finite difference methods.
This analysis can be used to predict convergance rates and optimize parameters in multilevel methods and preconditioners.

This package provides a toolkit for analyzing the performance of preconditioners for arbitrary, user provided weak forms of partial differential equations.

## Installing

To install, run

```
julia --project -e 'using Pkg; Pkg.build()'
```

To install and test, run

```
julia --project -e 'using Pkg; Pkg.build(); Pkg.test("LFAToolkit")'
```

