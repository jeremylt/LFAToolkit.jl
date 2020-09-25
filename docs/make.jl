using Documenter, DocumenterTools, LFAToolkit
DocMeta.setdocmeta!(LFAToolkit, :DocTestSetup, :(using LFAToolkit); recursive = true)

# The DOCSARGS environment variable can be used to pass additional arguments to make.jl.
# This is useful on CI, if you need to change the behavior of the build slightly but you
# can not change the .travis.yml or make.jl scripts any more (e.g. for a tag build).
if haskey(ENV, "DOCSARGS")
    for arg in split(ENV["DOCSARGS"])
        (arg in ARGS) || push!(ARGS, arg)
    end
end

makedocs(
    modules = [LFAToolkit],
    format = Documenter.HTML(
        # Use clean URLs, unless built as a "local" build
        prettyurls = !("local" in ARGS),
        canonical = "https://jeremylt.github.io/LFAToolkit.jl/stable/",
        highlights = ["yaml"],
    ),
    pages = [
        "Introduction" => "index.md",
        "Mathematical Background" => "background.md",
        "Public API" => [
            "Overview" => "public.md",
            "Rectangular Mesh" => "public/mesh.md",
            "Finite Element Basis" => "public/basis.md",
            "Basis Evaluation Mode" => "public/evaluationmode.md",
            "Operator Field" => "public/operatorfield.md",
            "Finite Element Operator" => "public/operator.md",
            "Preconditioner: Jacobi" => "public/pc_jacobi.md",
        ],
        "Private API" => [
            "Overview" => "private.md",
            "Rectangular Mesh" => "private/mesh.md",
            "Finite Element Basis" => "private/basis.md",
            "Finite Element Operator" => "private/operator.md",
            "Preconditioner: Jacobi" => "private/pc_jacobi.md",
        ],
        "References" => "references.md",
    ],
    clean = false,
    sitename = "LFAToolkit.jl",
    authors = "Jeremy L Thompson",
    linkcheck = !("skiplinks" in ARGS),
)

deploydocs(
    repo = "github.com/jeremylt/LFAToolkit.jl.git",
    target = "build",
    push_preview = true,
)
