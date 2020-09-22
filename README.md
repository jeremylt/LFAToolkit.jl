# LFAToolkit

[![travis](https://travis-ci.com/jeremylt/LFAToolkit.jl.svg?branch=master)](https://travis-ci.com/github/jeremylt/LFAToolkit.jl)
[![codecov](https://codecov.io/gh/jeremylt/LFAToolkit.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jeremylt/LFAToolkit.jl)
[![docs](https://img.shields.io/badge/docs-dev-blue)](https://jeremylt.github.io/LFAToolkit.jl/dev/)

Local Fourier Analysis for arbitrary order finite element type operators

## Introduction

Local Fourier Analysis is a tool commonly used in the analysis of multigrid and multilevel algorthms for solving partial differential equations via finite element or finite difference methods.
This analysis can be used to predict convergance rates and optimize parameters in multilevel methods and preconditioners.

This package provides a toolkit for analyzing the performance of preconditioners for arbitrary, user provided weak forms of partial differential equations.