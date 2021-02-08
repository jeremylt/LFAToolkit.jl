## Preconditioner: Chebyshev

### Example

This is an example of a Chebyshev smoother.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../../examples/ex111_chebyshev.jl", String))
```
""")
````

### Documentation

```@docs
Chebyshev
seteigenvalueestimatescaling
computesymbols(::Chebyshev, ::Array, ::Array)
```
