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

module FieldMode

"""
    Field evaluation mode for operator inputs and outputs

    Active - field represents unknowns or test functions
    Passive - field represents static data
"""
@enum FMode active passive

end # submodule

# ---------------------------------------------------------------------------------------------------------------------
