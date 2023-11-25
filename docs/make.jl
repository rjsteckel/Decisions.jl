using Decisions
using Documenter

DocMeta.setdocmeta!(Decisions, :DocTestSetup, :(using Decisions); recursive=true)

makedocs(;
    modules=[Decisions],
    authors="Ryan Steckel <rsteckel@gmail.com> and contributors",
    repo="https://github.com/rjsteckel/Decisions.jl/blob/{commit}{path}#{line}",
    sitename="Decisions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://rjsteckel.github.io/Decisions.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/rjsteckel/Decisions.jl",
    devbranch="main",
)
