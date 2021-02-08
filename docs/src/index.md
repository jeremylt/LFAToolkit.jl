# LFAToolkit

Local Fourier Analysis for arbitrary order finite element type operators

## Introduction

Local Fourier Analysis is a tool commonly used in the analysis of multigrid and multilevel algorithms for solving partial differential equations via finite element or finite difference methods.
This analysis can be used to predict convergence rates and optimize parameters in multilevel methods and preconditioners.

This package provides a toolkit for analyzing the performance of preconditioners for arbitrary, user provided weak forms of partial differential equations.

![](img/multi_grid_spectral_radius_9_to_2_2d.png)

Example plot for the symbol of p-multigrid with a Jacobi smoother for the 2D scalar diffusion problem.

## Contents

```@contents
Pages = [
    "background.md",
    "public.md",
    "private.md",
    "references.md"
]
Depth = 1
```

