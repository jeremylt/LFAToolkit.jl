# ---------------------------------------------------------------------------------------------------------------------
# Operator fields
# ---------------------------------------------------------------------------------------------------------------------

"""
    OperatorField(
        basis,
        evaluationmodes
    )

Finite Element operator input or output, with a basis and evaluation mode
"""
struct OperatorField
    basis::Basis
    evaluationmodes::Array{EvaluationMode.EvalMode}
end

# ---------------------------------------------------------------------------------------------------------------------
