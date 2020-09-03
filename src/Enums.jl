# ---------------------------------------------------------------------------------------------------------------------
# Enumerated types
# ---------------------------------------------------------------------------------------------------------------------

module EvaluationMode

"""
    Basis evaluation mode for operator inputs and outputs

    Interpolation - values interpolated to quadrature points
    Gradient - derivatives evaluated at quadrature points
    Quadrature Weights - quadrature weights
"""
@enum EvalMode interpolation gradient quadratureweights

end # submodule

# ---------------------------------------------------------------------------------------------------------------------
