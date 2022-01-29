using DeepCFR
using Test
using CounterfactualRegret

const CFR = CounterfactualRegret
using StaticArrays

function DeepCFR.vectorized(game::CFR.Kuhn, I)
    p, pc, hist = I
    L = length(hist)
    h = Tuple(hist)
    leftover = ntuple(_ -> -1, 2 - L)
    SA[p, pc, h..., leftover...]::SVector{4,Int}
end

@testset "DeepCFR.jl" begin
    game = CFR.Kuhn()
    sol = DeepCFRSolver(game)
    train!(sol, 10)
    I0 = infokey(game, initialhist(game))
    σ = @inferred sol(DeepCFR.vectorized(game, I0))
    @test σ isa Vector{<:AbstractFloat}
    @test sum(σ) ≈ 1.0
end
