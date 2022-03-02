using DeepCFR
using Test
using CounterfactualRegret
const CFR = CounterfactualRegret
using StaticArrays
using Flux

DeepCFR.vectorized(::CFR.IIEMatrixGame, I) = SA[I]

function DeepCFR.vectorized(game::CFR.Kuhn, I)
    p, pc, hist = I
    L = length(hist)
    h = Tuple(hist)
    leftover = ntuple(_ -> -1, 2 - L)
    SA[p, pc, h..., leftover...]::SVector{4,Int}
end

@testset "Default Networks" begin
    game = CFR.Kuhn()
    sol = DeepCFRSolver(game)
    train!(sol, 10)
    I0 = infokey(game, initialhist(game))
    σ = @inferred sol(DeepCFR.vectorized(game, I0))
    @test σ isa Vector{<:AbstractFloat}
    @test sum(σ) ≈ 1.0

    game = CFR.IIEMatrixGame()
    sol = DeepCFRSolver(game)
    train!(sol, 10)
    I0 = infokey(game, initialhist(game))
    σ = @inferred sol(DeepCFR.vectorized(game, I0))
    @test σ isa Vector{<:AbstractFloat}
    @test sum(σ) ≈ 1.0
end

@testset "userdef" begin

    σ = Chain(Dense(1,20, sigmoid), Dense(20,20, relu), Dense(20,3), softmax)
    v_net = Chain(Dense(1,20, sigmoid), Dense(20,20, relu), Dense(20,3))

    game = CFR.IIEMatrixGame()
    sol = DeepCFRSolver(
        game;
        buffer_size = 100*10^3,
        batch_size = 128,
        traversals = 10,
        strategy = σ,
        values = (v_net, deepcopy(v_net)),
    )
    train!(sol, 10)
    I0 = infokey(game, initialhist(game))
    σ = @inferred sol(DeepCFR.vectorized(game, I0))
    @test σ isa Vector{<:AbstractFloat}
    @test sum(σ) ≈ 1.0
end
