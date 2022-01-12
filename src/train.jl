#=
https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/cfr/deep_cfr.jl
=#
function CounterfactualRegret.train!(sol::DeepCFRSolver, N::Int)
    initialize!(sol)
    h0 = initialhist(sol.game)
    t = 0
    for _ in 1:N
        for _ in 1:sol.batch_size
            t += 1
            for p in 1:2
                traverse(sol, h0, p, t)
            end
        end
        train_value!(sol, 1)
        train_value!(sol, 2)
    end
    train_policy!(sol)
end

"""
reset advantage network and empty memory buffers
"""
function initialize!(sol::DeepCFRSolver)
    # https://discourse.julialang.org/t/reset-model-parameters-flux-jl/35021/2
    Flux.loadparams!(sol.V[1], map(p -> p .= randn.(), Flux.params(sol.V[1])))
    Flux.loadparams!(sol.V[2], map(p -> p .= randn.(), Flux.params(sol.V[2])))
    # TODO: Reinitialize with glorot somehow? Or some user-specified initializer?
    # TODO: empty memory buffers
end

function train_value!(sol::DeepCFRSolver, p::Int)
    train_net!(sol.V[p], sol.Mv[p].I, sol.Mv[p].r, sol.batch_size, sol.optimizer)
end

function train_policy!(sol::DeepCFRSolver)
    train_net!(sol.Π, sol.Mπ.I, sol.Mπ.σ, sol.batch_size, sol.optimizer)
end

function train_net!(net, x_data, y_data, batch_size, opt)
    L = length(x_data)
    iszero(L) && return
    full_batches, leftover = divrem(L, batch_size)
    total_batches = full_batches
    !iszero(leftover) && (total_batches += 1)

    input_size = length(first(x_data))
    output_size = length(first(y_data))
    perm = randperm(L)
    perms = Flux.chunk(perm, total_batches)

    X = Matrix{Float64}(undef, input_size, batch_size)
    Y = Matrix{Float64}(undef, output_size, batch_size)

    for i in 1:full_batches
        fillmat!(X::Matrix{Float64}, x_data[perms[i]])
        fillmat!(Y::Matrix{Float64}, y_data[perms[i]])
        p = params(net)
        gs = gradient(p) do
            Flux.Losses.mse(net(X), Y)
        end
        Flux.update!(opt, p, gs)
    end

    if !iszero(leftover)
        X = Matrix{Float64}(undef, input_size, leftover)
        Y = Matrix{Float64}(undef, output_size, leftover)

        fillmat!(X::Matrix{Float64}, x_data[last(perms)])
        fillmat!(Y::Matrix{Float64}, y_data[last(perms)])
        p = params(net)
        gs = gradient(p) do
            Flux.Losses.mse(net(X),Y)
        end
        Flux.update!(opt, p, gs)
    end

    nothing
end

"""
inplace `reduce(hcat, vecvec)`
"""
function fillmat!(mat::Matrix{T}, vecvec) where T
    inner_sz, outer_sz = size(mat)
    @assert length(vecvec)==outer_sz "$(length(vecvec)) ≠ $outer_sz"
    @assert length(first(vecvec)) == inner_sz "$(length(first(vecvec))) ≠ $inner_sz"
    for i in eachindex(vecvec)
        mat[:,i] .= vecvec[i]
    end
    mat
end
