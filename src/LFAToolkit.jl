# ------------------------------------------------------------------------------
# Local Fourier Analysis Toolkit
# ------------------------------------------------------------------------------

module LFAToolkit

# ------------------------------------------------------------------------------
# standard libraries
# ------------------------------------------------------------------------------

using LinearAlgebra
using Printf
using SparseArrays

# ------------------------------------------------------------------------------
# user available types and methods
# ------------------------------------------------------------------------------

export EvaluationMode, MultigridType
# mesh
export Mesh1D, Mesh2D, Mesh3D
# bases
export TensorBasis,
    NonTensorBasis,
    TensorH1LagrangeBasis,
    TensorH1UniformBasis,
    TensorMacroElementBasisFrom1D,
    TensorH1LagrangeMacroBasis,
    TensorH1UniformMacroBasis,
    TensorHProlongationBasis,
    TensorH1LagrangeHProlongationBasis,
    TensorH1UniformHProlongationBasis,
    TensorH1LagrangePProlongationBasis,
    TensorHProlongationMacroBasisFrom1D,
    TensorH1LagrangeHProlongationMacroBasis,
    TensorH1UniformHProlongationMacroBasis
# operato oields
export OperatorField
# operators
export Operator, GalleryOperator, computesymbols
# preconditioners
export IdentityPC
export Jacobi
export Chebyshev, seteigenvalueestimatescaling
# -- multigrid
export Multigrid, PMultigrid, HMultigrid
# -- BDDC
export BDDC, LumpedBDDC, DirichletBDDC

# ------------------------------------------------------------------------------
# imports
# ------------------------------------------------------------------------------

include("Enums.jl")
# mesh
include("Mesh.jl")
# bases
include("Basis/Base.jl")
include("Basis/Constructors.jl")
# operator fields
include("OperatorField.jl")
# operators
include("Operator/Base.jl")
include("Operator/Constructors.jl")
# preconditioners
include("PC/Base.jl")
include("PC/Identity.jl")
include("PC/Jacobi.jl")
include("PC/Chebyshev.jl")
# -- multigrid
include("PC/Multigrid/Base.jl")
include("PC/Multigrid/Constructors.jl")
# -- BDDC
include("PC/BDDC/Base.jl")
include("PC/BDDC/Constructors.jl")

end # module

# ------------------------------------------------------------------------------
