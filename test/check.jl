@testset "Check" begin
    game = CFR.IIEMatrixGame()

    #=
    Wrong input size:
    First dense layer has dimension 2, but should be 1
    =#
    strat = Chain(Dense(2,20, sigmoid), Dense(20,20, relu), Dense(20,3), softmax)
    v_net = Chain(Dense(2,20, sigmoid), Dense(20,20, relu), Dense(20,3))

    sol = DeepCFRSolver(
        game;
        buffer_size = 100*10^3,
        batch_size = 128,
        traversals = 10,
        strategy = strat,
        values = (v_net, deepcopy(v_net)),
    )

    @test_throws DimensionMismatch DeepCFR.checksolver(sol)


    #=
    Wrong output size:
    output layer has dimension 4, but should be 3
    =#
    strat = Chain(Dense(1,20, sigmoid), Dense(20,20, relu), Dense(20,4), softmax)
    v_net = Chain(Dense(1,20, sigmoid), Dense(20,20, relu), Dense(20,4))


    sol = DeepCFRSolver(
        game;
        buffer_size = 100*10^3,
        batch_size = 128,
        traversals = 10,
        strategy = strat,
        values = (v_net, deepcopy(v_net)),
    )

    @test_throws AssertionError DeepCFR.checksolver(sol)

    #=
    Unnormalized strategy network output:
    sum of strategy network outputs != 1.0
    =#
    strat = Chain(Dense(1,20, sigmoid), Dense(20,20, relu), Dense(20,3))
    sol = DeepCFRSolver(
        game;
        buffer_size = 100*10^3,
        batch_size = 128,
        traversals = 10,
        strategy = strat,
    )

    @test_throws AssertionError DeepCFR.checksolver(sol)


    strat = Chain(Dense(1,20, sigmoid), Dense(20,20, relu), Dense(20,3), softmax)
    v_net = Chain(Dense(1,20, sigmoid), Dense(20,20, relu), Dense(20,3))

    sol = DeepCFRSolver(
        game;
        buffer_size = 100*10^3,
        batch_size = 128,
        traversals = 10,
        strategy = strat,
        values = (v_net, deepcopy(v_net)),
    )

    @test DeepCFR.checksolver(sol) == "Tests Pass"
end
