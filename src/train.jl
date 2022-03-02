#=
https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/cfr/deep_cfr.jl
=#
function CounterfactualRegret.train!(sol::DeepCFRSolver, N::Int; show_progress::Bool = false, cb = () -> ())
    cb = Flux.Optimise.runall(cb)
    initialize!(sol)
    h0 = initialhist(sol.game)
    prog = Progress(N; enabled=show_progress)
    for t in 1:N
        cb()
        for _ in 1:sol.traversals
            for p in 1:2
                traverse(sol, h0, p, sol.T)
            end
        end
        train_value!(sol, 1)
        train_value!(sol, 2)
        next!(prog)
        sol.T += 1.0f0
    end
    train_policy!(sol)
end

"""
https://discourse.julialang.org/t/reset-model-parameters-flux-jl/35021/2
"""
function initialize!(nn::Union{Dense, Chain}) # These networks don't have a common supertype?
    Flux.loadparams!(nn, map(p -> p .= randn.(Float32), Flux.params(nn)))
end

"""
reset advantage networks and empty memory buffers
"""
function initialize!(sol::DeepCFRSolver)
    # TODO: Reinitialize with glorot somehow? Or some user-specified initializer?
    # TODO: empty memory buffers
    initialize!.(sol.V)
    initialize!(sol.Π)
end

function train_value!(sol::DeepCFRSolver, p::Int)
    initialize!.(sol.V)
    opt = deepcopy(sol.advantage_opt)
    for _ in 1:sol.value_epochs
        train_net!(
            sol.gpu,
            sol.V[p],
            sol.Mv[p].I,
            sol.Mv[p].r,
            sol.Mv[p].t,
            sol.batch_size,
            opt
        )
    end
end

function train_policy!(sol::DeepCFRSolver, epochs::Int=sol.strategy_epochs, opt=sol.strategy_opt)
    for _ in 1:epochs
        train_net!(
            sol.gpu,
            sol.Π,
            sol.Mπ.I,
            sol.Mπ.σ,
            sol.Mπ.t,
            sol.batch_size,
            opt
        )
    end
end

function train_net!(::Val{true}, dest_net, x_data, y_data, w, batch_size, opt)
    src_net = dest_net |> gpu
    L = length(x_data)
    iszero(L) && return
    full_batches, leftover = divrem(L, batch_size)
    total_batches = full_batches
    !iszero(leftover) && (total_batches += 1)

    input_size = length(first(x_data))
    output_size = length(first(y_data))
    perm = randperm(L)
    perms = collect(Iterators.partition(perm, batch_size))
    p = params(src_net)

    _X = Matrix{Float32}(undef, input_size, batch_size)
    _Y = Matrix{Float32}(undef, output_size, batch_size)
    X = _X |> gpu
    Y = _Y |> gpu
    W = Vector{Float32}(undef, batch_size) |> gpu

    for i in 1:full_batches
        fillmat!(_X, @view x_data[perms[i]])
        fillmat!(_Y, @view y_data[perms[i]])
        copyto!(X, _X)
        copyto!(Y, _Y)
        copyto!(W, w[perms[i]])

        Loss = NetLoss(src_net, X, Y, W)

        gs = gradient(Loss, p)

        Flux.update!(opt, p::Flux.Params, gs)
    end

    if !iszero(leftover)
        _X = Matrix{Float32}(undef, input_size, leftover)
        _Y = Matrix{Float32}(undef, output_size, leftover)
        fillmat!(_X, @view x_data[last(perms)])
        fillmat!(_Y, @view y_data[last(perms)])
        X = _X |> gpu
        Y = _Y |> gpu
        W = w[last(perms)] |> gpu

        Loss = NetLoss(src_net, X, Y, W)
        gs = gradient(Loss, p)
        Flux.update!(opt, p::Flux.Params, gs)
    end

    Flux.loadparams!(dest_net, p)
    nothing
end

function train_net!(::Val{false}, net, x_data, y_data, w, batch_size, opt)
    L = length(x_data)
    iszero(L) && return
    full_batches, leftover = divrem(L, batch_size)
    total_batches = full_batches
    !iszero(leftover) && (total_batches += 1)

    input_size = length(first(x_data))
    output_size = length(first(y_data))
    perm = randperm(L)
    perms = collect(Iterators.partition(perm, batch_size))

    X = Matrix{Float32}(undef, input_size, batch_size)
    Y = Matrix{Float32}(undef, output_size, batch_size)
    W = Vector{Float32}(undef, batch_size)
    Loss = NetLoss(net, X, Y, W)
    p = params(net)

    for i in 1:full_batches
        fillmat!(X::Matrix{Float32}, @view x_data[perms[i]])
        fillmat!(Y::Matrix{Float32}, @view y_data[perms[i]])
        copyto!(W::Vector{Float32}, @view w[perms[i]])
        gs = gradient(Loss, p)

        Flux.update!(opt, p::Flux.Params, gs)
    end

    if !iszero(leftover)
        X = Matrix{Float32}(undef, input_size, leftover)
        Y = Matrix{Float32}(undef, output_size, leftover)
        W = w[last(perms)]
        Loss = NetLoss(net, X, Y, W)

        fillmat!(X::Matrix{Float32}, @view x_data[last(perms)])
        fillmat!(Y::Matrix{Float32}, @view y_data[last(perms)])
        gs = gradient(Loss, p)
        Flux.update!(opt, p::Flux.Params, gs)
    end

    nothing
end

train_net!(args...) = train_net!(Val(false), args...)

"""
inplace `reduce(hcat, vecvec)`
"""
function fillmat!(mat::T, vecvec) where T
    inner_sz, outer_sz = size(mat)
    @assert length(vecvec) == outer_sz "$(length(vecvec)) ≠ $outer_sz"
    @assert length(first(vecvec)) == inner_sz "$(length(first(vecvec))) ≠ $inner_sz"
    @inbounds for i in eachindex(vecvec)
        dest = view(mat,:,i)
        copyto!(dest, vecvec[i])
    end
    mat::T
end

"""
Weighted mean squared error
"""
wmse(ŷ,y,w) = sum(abs2.(ŷ .- y)*w ./ length(w))

struct NetLoss{NN,M,WGT}
    net::NN
    X::M
    Y::M
    W::WGT
end

(n::NetLoss)() = wmse(n.net(n.X), n.Y, n.W)
