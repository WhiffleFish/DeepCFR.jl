module DeepCFR

using CounterfactualRegret
using Flux

include("solver.jl")
include("ESCFR.jl")
include("train.jl")

export DeepCFRSolver
end
