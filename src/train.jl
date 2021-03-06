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
        for p in 1:2
            for _ in 1:sol.traversals
                sol.traverser(sol, h0, p, sol.T)
            end
            train_value!(sol, p)
        end
        next!(prog)
        sol.T += 1.0f0
    end
    train_policy!(sol)
end

function initialize!(nn, init=Flux.glorot_normal)
    for p in Flux.params(nn)
        p .= init(size(p)...)
    end
end

"""
reset advantage networks and empty memory buffers
"""
function initialize!(sol::DeepCFRSolver)
    initialize!.(sol.V)
    initialize!(sol.Π)
end

function train_value!(sol::DeepCFRSolver, p::Int)
    initialize!(sol.V[p])
    opt = deepcopy(sol.advantage_opt)
    train_net!(
        sol.gpu,
        sol.V[p],
        sol.Mv[p].I,
        sol.Mv[p].r,
        sol.Mv[p].t,
        sol.batch_size,
        sol.value_batches,
        opt
    )
end

function train_policy!(sol::DeepCFRSolver, batches::Int=sol.strategy_batches, opt=sol.strategy_opt)
    train_net!(
        sol.gpu,
        sol.Π,
        sol.Mπ.I,
        sol.Mπ.σ,
        sol.Mπ.t,
        sol.batch_size,
        sol.strategy_batches,
        opt
    )
end

function train_net!(
    ::Val{true},
    dest_net,
    x_data,
    y_data,
    w,
    batch_size,
    n_batches,
    opt)

    isempty(x_data) && return nothing
    src_net = dest_net |> gpu

    input_size = length(first(x_data))
    output_size = length(first(y_data))
    p = Flux.params(src_net)

    _X = Matrix{Float32}(undef, input_size, batch_size)
    _Y = Matrix{Float32}(undef, output_size, batch_size)
    sample_idxs = Vector{Int}(undef, batch_size)
    idxs = 1:length(w)

    X = _X |> gpu
    Y = _Y |> gpu
    W = Vector{Float32}(undef, batch_size) |> gpu

    for i in 1:n_batches
        rand!(sample_idxs, idxs)
        fillmat!(_X, x_data, sample_idxs)
        fillmat!(_Y, y_data, sample_idxs)
        copyto!(X, _X)
        copyto!(Y, _Y)
        copyto!(W, w[sample_idxs])

        Loss = NetLoss(src_net, X, Y, W)

        gs = gradient(Loss, p)

        Flux.update!(opt, p::Flux.Params, gs)
    end

    Flux.loadmodel!(dest_net, src_net)
    nothing
end

function train_net!(
    ::Val{false},
    net,
    x_data,
    y_data,
    w,
    batch_size,
    n_batches,
    opt)

    isempty(x_data) && return nothing
    input_size = length(first(x_data))
    output_size = length(first(y_data))

    X = Matrix{Float32}(undef, input_size, batch_size)
    Y = Matrix{Float32}(undef, output_size, batch_size)
    W = Vector{Float32}(undef, batch_size)
    sample_idxs = Vector{Int}(undef, batch_size)
    idxs = 1:length(w)

    Loss = NetLoss(net, X, Y, W)
    p = Flux.params(net)

    for i in 1:n_batches
        rand!(sample_idxs, idxs)
        fillmat!(X::Matrix{Float32}, x_data, sample_idxs)
        fillmat!(Y::Matrix{Float32}, y_data, sample_idxs)
        copyto!(W::Vector{Float32}, @view w[sample_idxs])

        gs = gradient(Loss, p)

        Flux.update!(opt, p::Flux.Params, gs)
    end

    nothing
end

train_net!(args...) = train_net!(Val(false), args...)

function fillmat!(mat::AbstractMatrix, vecvec::AbstractVector, idxs)
    @inbounds for i in 1:size(mat, 2)
        mat[:,i] .= vecvec[idxs[i]]
    end
    return mat
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
