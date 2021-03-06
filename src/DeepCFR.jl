module DeepCFR

using CounterfactualRegret
const CFR = CounterfactualRegret
using Random
using ProgressMeter
using Flux
using FileIO
using RecipesBase

include("memory.jl")
include("solver.jl")
include("ESCFR.jl")
include("OSCFR.jl")
include("train.jl")
include("check.jl")
include("callback.jl")
include("fitting.jl")

export DeepCFRSolver

end
