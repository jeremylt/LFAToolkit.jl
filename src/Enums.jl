# ------------------------------------------------------------------------------
# enumerated types
# ------------------------------------------------------------------------------

module EvaluationMode

"""
Basis evaluation mode for operator inputs and outputs

# Modes:
- `interpolation`:     values interpolated to quadrature points
- `gradient`:          derivatives evaluated at quadrature points
- `quadratureweights`: quadrature weights

# Example:
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

module MultigridType

"""
Multigrid types

# Types:
- `pmultigrid`: p-multigrid
- `hmultigrid`: h-multigrid

# Example:
```@jldoctest
LFAToolkit.MultigridType.MgridType

# output
Enum LFAToolkit.MultigridType.MgridType:
pmultigrid = 0
hmultigrid = 1
```
"""
@enum MgridType hmultigrid pmultigrid

end # submodule

module BDDCInjectionType

"""
BDDC injection types

# Types:
- `scaled`:   scaled injection
- `harmonic`: discrete harmonic extension

# Example
```@jldoctest
LFAToolkit.BDDCInjectionType.BDDCInjectType

# output
Enum LFAToolkit.BDDCInjectionType.BDDCInjectType:
scaled = 0
harmonic = 1
```
"""
@enum BDDCInjectType scaled harmonic

end # submodule

# ------------------------------------------------------------------------------
