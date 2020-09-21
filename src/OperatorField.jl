# ---------------------------------------------------------------------------------------------------------------------
# operator fields
# ---------------------------------------------------------------------------------------------------------------------

"""
```julia
OperatorField(
    basis,
    evaluationmodes
)
```

Finite Element operator input or output, with a basis and evaluation mode

# Arguments:
- `basis`:           finite element basis for the field
- `evaluationmodes`: array of basis evaluation modes,
                     note that quadrature weights must be listed in a separate operator field

# Returns:
- Finite element operator field object

# Example:
```jldoctest
# basis
basis = TensorH1LagrangeBasis(4, 3, 2);

# quadrature weights field
weightsfield = OperatorField(basis, [EvaluationMode.quadratureweights]);

# verify
@assert weightsfield.basis == basis
@assert weightsfield.evaluationmodes[1] == EvaluationMode.quadratureweights

# input or output field
inputfield = OperatorField(basis, [
    EvaluationMode.interpolation,
    EvaluationMode.gradient,
]);

# verify
@assert inputfield.basis == basis
@assert inputfield.evaluationmodes[1] == EvaluationMode.interpolation
@assert inputfield.evaluationmodes[2] == EvaluationMode.gradient

# output

```
"""
struct OperatorField
    # data
    basis::Basis
    evaluationmodes::Array{EvaluationMode.EvalMode}

    # inner constructor
    OperatorField(basis, evaluationmodes) = (
        # validity checking
        if length(evaluationmodes) > 1 &&
           EvaluationMode.quadratureweights in evaluationmodes
            error("quadrature weights must be a separate operator field") # COV_EXCL_LINE
        end;

        # constructor
        new(basis, evaluationmodes)
    )
end

# ---------------------------------------------------------------------------------------------------------------------
