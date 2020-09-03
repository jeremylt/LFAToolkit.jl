using Test, LFAToolkit

# ---------------------------------------------------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------------------------------------------------

p = 4
q = 4
dimension = 1
numbercomponents = 1
nodes = [-1.0, -sqrt(5.0) / 5.0, sqrt(5.0) / 5.0, 1.0]
quadraturepoints = [-0.86113631, -0.33998104, 0.33998104, 0.86113631]
quadratureweights = [0.34785485, 0.65214515, 0.65214515, 0.34785485]
interpolation = [
    0.62994317 0.47255875 -0.14950343 0.04700152
    -0.07069480 0.97297619 0.13253993 -0.03482132
    -0.03482132 0.13253993 0.97297619 -0.07069480
    0.04700152 -0.14950343 0.47255875 0.62994317
]
gradient = [
    -2.34183742 2.78794489 -0.63510411 0.18899664
    -0.51670214 -0.48795249 1.33790510 -0.33325047
    0.33325047 -1.33790510 0.48795249 0.51670214
    -0.18899664 0.63510411 -2.78794489 2.34183742
]

basis = TensorH1LagrangeBasis(4, 4, 1, 1)

@test basis.p == p
@test basis.q == q
@test basis.dimension == dimension
@test basis.numbercomponents == numbercomponents
@test basis.istensor == true

tol = 1e-7
for i = 1:p
    @test abs(basis.nodes[i] - nodes[i]) < tol
end
for i = 1:q
    @test abs(basis.quadraturepoints[i] - quadraturepoints[i]) < tol
end
for i = 1:q, j = 1:p
    @test abs(basis.interpolation[i, j] - interpolation[i, j]) < tol
    @test abs(basis.gradient[i, j] - gradient[i, j]) < tol
end

# ---------------------------------------------------------------------------------------------------------------------
# Operator construction
# ---------------------------------------------------------------------------------------------------------------------

inputs = [OperatorField(basis, EvaluationMode.interpolation)]
outputs = [OperatorField(basis, EvaluationMode.interpolation)]

# ---------------------------------------------------------------------------------------------------------------------
