module DeepCFR

using CounterfactualRegret
using Random
using Flux

include("memory.jl")
include("solver.jl")
include("ESCFR.jl")
include("train.jl")

export DeepCFRSolver

end
