# LFAToolkit

Local Fourier Analysis for arbitrary order finite element type operators

## Introduction

Local Fourier Analysis (LFA) is a tool commonly used in the analysis of multigrid and multilevel algorithms for solving partial differential equations via finite element or finite difference methods.
This analysis can be used to predict convergence rates and optimize parameters in multilevel methods and preconditioners.

This library provides a toolkit for analyzing the performance of preconditioners for arbitrary, user provided weak forms of partial differential equations.
While this library focuses on the finite element discretizations, finite difference discretizations of PDEs can often be recovered from finite element formulations by using linear finite elements on a structured grid.
This fact makes LFAToolkit.jl an extremely flexible tool for LFA.

![](img/multi_grid_spectral_radius_9_to_2_2d.png)

Example plot for the symbol of p-multigrid with a Jacobi smoother for the 2D scalar diffusion problem.

## Contents

```@contents
Pages = [
    "background.md",
    "examples.md",
    "public.md",
    "private.md",
    "references.md"
]
Depth = 1
```