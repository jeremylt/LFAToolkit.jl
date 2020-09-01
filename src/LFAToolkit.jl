module LFAToolkit

export greet

greet() = "Hello World!"

include("Enums.jl")
include("Basis.jl")
include("OperatorField.jl")
include("Operator.jl")

end # module
