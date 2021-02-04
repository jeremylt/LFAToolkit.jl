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

export EvaluationMode
export Mesh1D, Mesh2D, Mesh3D
export TensorBasis, NonTensorBasis, TensorH1LagrangeBasis, TensorH1UniformBasis
export OperatorField
export Operator, GalleryOperator, computesymbols
export Jacobi
export Chebyshev, seteigenvalueestimatescaling
export PMultigrid

# ------------------------------------------------------------------------------
# imports
# ------------------------------------------------------------------------------

include("Enums.jl")
include("Mesh.jl")
include("Basis.jl")
include("OperatorField.jl")
include("Operator.jl")
include("PC_Base.jl")
include("PC_Jacobi.jl")
include("PC_Chebyshev.jl")
include("PC_PMultigrid.jl")

end # module

# ------------------------------------------------------------------------------
