# ---------------------------------------------------------------------------------------------------------------------
# Enumerated types
# ---------------------------------------------------------------------------------------------------------------------

module EvaluationMode

"""
Basis evaluation mode for operator inputs and outputs

Interpolation - values interpolated to quadrature points

Gradient - derivatives evaluated at quadrature points

Quadrature Weights - quadrature weights

```@jldoctest
EvaluationMode.EvalMode

# output
Enum LFAToolkit.EvaluationMode.EvalMode:
interpolation = 0
gradient = 1
quadratureweights = 2
```
"""
@enum EvalMode interpolation gradient quadratureweights

end # submodule

# ---------------------------------------------------------------------------------------------------------------------
