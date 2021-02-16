# ------------------------------------------------------------------------------
# documentation
# ------------------------------------------------------------------------------
using Documenter, DocumenterTools, Markdown, LFAToolkit
DocMeta.setdocmeta!(LFAToolkit, :DocTestSetup, :(using LFAToolkit); recursive = true)

# The DOCSARGS environment variable can be used to pass additional arguments to make.jl.
# This is useful on CI, if you need to change the behavior of the build slightly but you
# can not change the .travis.yml or make.jl scripts any more (e.g. for a tag build).
if haskey(ENV, "DOCSARGS")
    for arg in split(ENV["DOCSARGS"])
        (arg in ARGS) || push!(ARGS, arg)
    end
end

# ------------------------------------------------------------------------------
# make
# ------------------------------------------------------------------------------
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
        "Examples" => "examples.md",
        "Public API" => "public.md",
        "Private API" => "private.md",
        "References" => "references.md",
    ],
    clean = false,
    sitename = "LFAToolkit.jl",
    authors = "Jeremy L Thompson",
    linkcheck = !("skiplinks" in ARGS),
)

# ------------------------------------------------------------------------------
# deploy
# ------------------------------------------------------------------------------
deploydocs(
    repo = "github.com/jeremylt/LFAToolkit.jl.git",
    devbranch = "main",
    target = "build",
    push_preview = true,
)

# ------------------------------------------------------------------------------
