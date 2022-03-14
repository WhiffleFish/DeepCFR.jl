@testset "Memory" begin
    game = CFR.Kuhn()
    sol = DeepCFRSolver(game; buffer_size=50, traversals=10)
    train!(sol, 100)

    @test length(sol.Mπ) == length(sol.Mπ.I) == length(sol.Mπ.t) == length(sol.Mπ.σ)
    @test length(sol.Mπ) == sol.Mπ.capacity == 50

    Mv1 = first(sol.Mv)
    Mv2 = first(sol.Mv)
    @test length(Mv1) == length(Mv1.I) == length(Mv1.t) == length(Mv1.r)
    @test length(Mv1) == Mv1.capacity == 50
    @test length(Mv2) == length(Mv2.I) == length(Mv2.t) == length(Mv2.r)
    @test length(Mv2) == Mv2.capacity == 50

    empty!(sol.Mπ)
    @test length(sol.Mπ) == 0
    empty!(Mv1)
    @test length(Mv1) == 0
end
