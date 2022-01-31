function total_value_loss(nn, Mv)
    L = length(Mv)
    input_size = length(first(Mv.I))
    output_size = length(first(Mv.r))
    X = Matrix{Float64}(undef, input_size, L)
    Y = Matrix{Float64}(undef, output_size, L)
    DeepCFR.fillmat!(X, Mv.I)
    DeepCFR.fillmat!(Y, Mv.r)
    return Flux.mse(nn(X),Y)
end

function total_strategy_loss(nn, Mπ)
    L = length(Mπ)
    input_size = length(first(Mπ.I))
    output_size = length(first(Mπ.σ))
    X = Matrix{Float64}(undef, input_size, L)
    Y = Matrix{Float64}(undef, output_size, L)
    DeepCFR.fillmat!(X, Mπ.I)
    DeepCFR.fillmat!(Y, Mπ.σ)
    return Flux.mse(nn(X),Y)
end

function monitored_train_value!(sol::DeepCFRSolver, p::Int)
    l = zeros(Float64, sol.value_epochs + 1)
    DeepCFR.initialize!.(sol.V)
    l[1] = total_value_loss(sol.V[p], sol.Mv[p])
    for i in 1:sol.value_epochs
        DeepCFR.train_net!(sol.V[p], sol.Mv[p].I, sol.Mv[p].r, sol.batch_size, sol.advantage_opt)
        l[i+1] = total_value_loss(sol.V[p], sol.Mv[p])
    end
    return l
end

function monitored_train_strategy!(sol::DeepCFRSolver)
    opt = ADAM()
    l = zeros(Float64, sol.strategy_epochs + 1)
    DeepCFR.initialize!(sol.Π)
    l[1] = total_strategy_loss(sol.Π, sol.Mπ)
    for i in 1:sol.strategy_epochs
        DeepCFR.train_net!(sol.Π, sol.Mπ.I, sol.Mπ.σ, sol.batch_size, opt)
        l[i+1] = total_strategy_loss(sol.Π, sol.Mπ)
    end
    return l
end

function value_consistency!(sol::DeepCFRSolver, N::Int; show_progress::Bool = true)
    loss_hist1 = Vector{Float64}[]
    loss_hist2 = Vector{Float64}[]
    strat_loss_hist = Vector{Float64}[]
    DeepCFR.initialize!(sol)
    h0 = initialhist(sol.game)
    t = 0
    prog = Progress(N; enabled=show_progress)
    for _ in 1:N
        for _ in 1:sol.traversals
            t += 1
            for p in 1:2
                DeepCFR.traverse(sol, h0, p, t)
            end
        end
        l1 = monitored_train_value!(sol, 1)
        l2 = monitored_train_value!(sol, 2)
        l3 = monitored_train_strategy!(sol)
        push!(loss_hist1, l1)
        push!(loss_hist2, l2)
        push!(strat_loss_hist, l3)
        next!(prog, showvalues=[(:MSE, last(l3))])
    end
    DeepCFR.train_policy!(sol)
    return loss_hist1, loss_hist2, strat_loss_hist
end
