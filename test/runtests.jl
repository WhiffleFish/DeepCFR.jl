using DeepCFR
using Test
using CounterfactualRegret
using CounterfactualRegret.Games
const CFR = CounterfactualRegret
using StaticArrays
using JLD2
using Flux
using Random

Random.seed!(1337)

include(joinpath(@__DIR__, "solver.jl"))

include(joinpath(@__DIR__, "fitting.jl"))

include(joinpath(@__DIR__, "memory.jl"))

include(joinpath(@__DIR__, "check.jl"))

include(joinpath(@__DIR__, "callback.jl"))
