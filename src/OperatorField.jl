# ---------------------------------------------------------------------------------------------------------------------
# Operator fields
# ---------------------------------------------------------------------------------------------------------------------

"""
    OperatorField(
        basis,
        evaluationmode
    )

Finite Element operator input or output, with a basis and evaluation mode
"""
struct OperatorField
    basis::Basis
    evaluationmode::EvaluationMode.EvalMode
end

# ---------------------------------------------------------------------------------------------------------------------
