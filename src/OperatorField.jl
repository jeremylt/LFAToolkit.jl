# ---------------------------------------------------------------------------------------------------------------------
# Operator fields
# ---------------------------------------------------------------------------------------------------------------------

"""
    OperatorField(
        basis,
        evaluationmodes
    )

Finite Element operator input or output, with a basis and evaluation mode

# Arguments:
- `basis`:           finite element basis for the field
- `evaluationmodes`: array of basis evaluation modes,
                     note that quadrature weights must be listed in a separate operator field

# Returns:
- Finite element operator field object
"""
struct OperatorField
    basis::Basis
    evaluationmodes::Array{EvaluationMode.EvalMode}
end

# ---------------------------------------------------------------------------------------------------------------------
