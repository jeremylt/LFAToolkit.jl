# ------------------------------------------------------------------------------
# Local Fourier Analysis Toolkit
# ------------------------------------------------------------------------------

module LFAToolkit

# ------------------------------------------------------------------------------
# standard libraries
# ------------------------------------------------------------------------------

using SparseArrays
using LinearAlgebra

# ------------------------------------------------------------------------------
# user available types and methods
# ------------------------------------------------------------------------------

export EvaluationMode
export Mesh1D, Mesh2D, Mesh3D
export TensorBasis, NonTensorBasis, TensorH1LagrangeBasis
export OperatorField
export Operator, computesymbols
export Jacobi

# ------------------------------------------------------------------------------
# imports
# ------------------------------------------------------------------------------

include("Enums.jl")
include("Mesh.jl")
include("Basis.jl")
include("OperatorField.jl")
include("Operator.jl")
include("PC_Jacobi.jl")

end # module

# ------------------------------------------------------------------------------
