using DeepCFR
using Test
using CounterfactualRegret
const CFR = CounterfactualRegret
using StaticArrays
using JLD2
using Flux

include(joinpath(@__DIR__, "solver.jl"))

include(joinpath(@__DIR__, "memory.jl"))

include(joinpath(@__DIR__, "check.jl"))

include(joinpath(@__DIR__, "callback.jl"))
