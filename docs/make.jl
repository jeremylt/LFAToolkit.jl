using Documenter, DocumenterTools, LFAToolkit

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
