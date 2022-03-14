DeepCFR.vectorized(::CFR.IIEMatrixGame, I) = SA[Float32(I)]

function DeepCFR.vectorized(game::CFR.Kuhn, I)
    p, pc, hist = I
    h = convert(SVector{3,Float32}, hist)
    SA[Float32(p), Float32(pc), h...]
end

@testset "Solver" begin

    ## Basic operation + reasonable outputs
    game = CFR.Kuhn()
    sol = DeepCFRSolver(game; buffer_size=50, traversals=10)
    train!(sol, 100)
    I0 = infokey(game, initialhist(game))
    σ = @inferred sol(DeepCFR.vectorized(game, I0))

    @test σ isa Vector{<:AbstractFloat}
    @test sum(σ) ≈ 1.0
    @test DeepCFR.on_gpu(sol) == false
    @test DeepCFR.infotype(sol) == SVector{5,Float32}
    @test DeepCFR.regrettype(sol) == Vector{Float32}
    @test DeepCFR.strattype(sol) == Vector{Float32}

    game = CFR.IIEMatrixGame()
    sol = DeepCFRSolver(game)
    train!(sol, 10)
    I0 = infokey(game, initialhist(game))
    σ = @inferred sol(DeepCFR.vectorized(game, I0))
    @test σ isa Vector{<:AbstractFloat}
    @test sum(σ) ≈ 1.0
    @test DeepCFR.on_gpu(sol) == false
    @test DeepCFR.infotype(sol) == SVector{1,Float32}
    @test DeepCFR.regrettype(sol) == Vector{Float32}
    @test DeepCFR.strattype(sol) == Vector{Float32}

    game = CFR.IIEMatrixGame()
    sol = DeepCFRSolver(game, on_gpu=true)
    train!(sol, 10)
    I0 = infokey(game, initialhist(game))
    σ = @inferred sol(DeepCFR.vectorized(game, I0))
    @test σ isa Vector{<:AbstractFloat}
    @test sum(σ) ≈ 1.0
    @test DeepCFR.on_gpu(sol) == true

    ## Printing

    io = IOBuffer()
    m = MIME"text/plain"()
    Base.show(io, m, sol)
    str = String(take!(io))
    @test occursin("GPU", str)
    @test occursin("Advantage Net", str)
    @test occursin("Advantage Memory", str)
    @test occursin("Strategy Net", str)
    @test occursin("Strategy Memory", str)
    @test occursin("Batch Size", str)
    @test occursin("Traversals", str)
    @test occursin("Advantage Optimizer", str)
    @test occursin("Strategy Optimizer", str)
    @test occursin("Game", str)
end
