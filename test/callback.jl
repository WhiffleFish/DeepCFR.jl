@testset "Callback" begin
    ## Throttle
    game = CFR.IIEMatrixGame()
    sol = DeepCFRSolver(game)
    io = IOBuffer()
    f = () -> println(io, length(sol.Mπ))
    save_freq = 10
    train_iter = 101
    cb = DeepCFR.Throttle(f,save_freq)
    train!(sol, train_iter, cb = cb)
    str = String(take!(io))
    @test length(split(str, "\n")) == div(train_iter, save_freq) + 2

    ## Model Saver
    sol = DeepCFRSolver(game)
    path = tempname(tempdir(); cleanup=true)
    cb = DeepCFR.ModelSaver(sol, save_freq, path=path)
    train!(sol, train_iter; cb = cb)

    file_list = readdir(path)
    @test length(file_list) == div(train_iter, save_freq) + 1
    model_path = joinpath(path,last(file_list))
    m = JLD2.load(model_path)["model"]
    @test typeof(m) == typeof(sol)

    ## Exploitability
    sol = DeepCFRSolver(game)
    cb = DeepCFR.ExploitabilityCallback(sol, 10)
    train!(sol, train_iter; cb = cb)
    @test length(cb.hist.y) == length(cb.hist.y) == 11
    @test first(cb.hist.y) > last(cb.hist.y) > 0.0
end
