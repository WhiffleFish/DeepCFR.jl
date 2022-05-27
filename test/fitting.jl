@testset "Fitting" begin
    game = Kuhn()

    sol = DeepCFRSolver(game)
    train!(sol, 100)

    p = 1
    net = sol.V[p]

    DeepCFR.initialize!(net, Flux.glorot_normal)

    dv_0 = DeepCFR.optimality_distance(net, sol.Mv[p])

    hist_glorot = DeepCFR.train_net_tracked!(
        net,
        sol.Mv[p],
        256,
        1000,
        ADAM(1e-3);
        show_progress = false
    )
    dv_f = DeepCFR.optimality_distance(net, sol.Mv[p])

    @test dv_0 > dv_f

    DeepCFR.initialize!(sol.Π, Flux.glorot_normal)
    dπ_0 = DeepCFR.optimality_distance(sol.Π, sol.Mπ)
    strat_hist = DeepCFR.train_net_tracked!(
        sol.Π,
        sol.Mπ,
        256,
        500,
        ADAM(1e-3);
        show_progress = false
    )

    dπ_f = DeepCFR.optimality_distance(sol.Π, sol.Mπ)

    @test dπ_0 > dπ_f
    @test last(hist_glorot) < first(hist_glorot)
end
