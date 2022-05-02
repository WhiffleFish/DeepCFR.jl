module DeepCFR

using CounterfactualRegret
const CFR = CounterfactualRegret
using Random
using ProgressMeter
using Flux
using FileIO

include("memory.jl")
include("solver.jl")
include("ESCFR.jl")
include("train.jl")
include("check.jl")
include("callback.jl")
include("fitting.jl")

export DeepCFRSolver

end
