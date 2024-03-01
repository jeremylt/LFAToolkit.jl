# LFAToolkit

[![CI Status](https://github.com/jeremylt/LFAToolkit.jl/actions/workflows/test.yml/badge.svg)](https://github.com/jeremylt/LFAToolkit.jl/actions)
[![CodeCov](https://codecov.io/gh/jeremylt/LFAToolkit.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jeremylt/LFAToolkit.jl)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Documentation](https://img.shields.io/badge/docs-dev-blue)](https://jeremylt.github.io/LFAToolkit.jl/dev/)
[![Documentation](https://img.shields.io/badge/docs-stable-blue)](https://jeremylt.github.io/LFAToolkit.jl/stable/)
[![DOI](https://zenodo.org/badge/291842028.svg)](https://zenodo.org/badge/latestdoi/291842028)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jeremylt/LFAToolkit.jl/main?filepath=examples%2Fjupyter)
[![Run on Repl.it](https://img.shields.io/badge/launch-replit-579aca?logo=replit)](https://replit.com/@jeremylt/LFAToolkitjl)

Local Fourier Analysis for arbitrary order finite element type operators

## Introduction

Local Fourier Analysis is a tool commonly used in the analysis of multigrid and multilevel algorithms for solving partial differential equations via finite element or finite difference methods.
This analysis can be used to predict convergence rates and optimize parameters in multilevel methods and preconditioners.

This package provides a toolkit for analyzing the performance of preconditioners for arbitrary, user provided weak forms of partial differential equations.

## Installing

To install a development version, run

```
$ julia --project -e 'using Pkg; Pkg.build();'
```

To install and test, run

```
$ julia --project -e 'using Pkg; Pkg.build(); Pkg.test("LFAToolkit")'
```

## Examples

Examples can be found in the ``examples`` directory, with interactive examples in Jupyter notebooks found in the ``examples/jupyter`` directory.

## Documentation

Documentation can be found at the url listed above.
To build the documentation locally, run

```
$ julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); Pkg.build()'
```

followed by

```
$ DOCSARGS=local julia --project=docs/ docs/make.jl
```

## Online Demos

Both of the current online demos are not performing smoothly.

The Binder with interactive Jupyter notebooks takes a long time to start due to the time required to install LFAToolkit.jl and the plotting utilities.

The repl.it often reports a System Error with `Disk quota exceeded`.
Clicking 'Run' again will typically clear the issue within a few attempts.

## Contact

You can reach the LFAToolkit.jl by leaving a comment in the [issue tracker](https://github.com/jeremylt/LFAToolkit.jl/issues).

## How to Cite

If you utilize LFAToolkit.jl please cite:

```bibtex
@software{LFAToolkit_jl,
  author  = {Thompson, Jeremy L and Bankole, Adeleke O and Brown, Jed},
  title   = {{LFAToolkit.jl}},
  version = {0.7.0},
  month   = mar,
  year    = {2024},
  url     = {https://jeremylt.github.io/LFAToolkit.jl/stable/},
  doi     = {10.5281/zenodo.4659283},
}
```
