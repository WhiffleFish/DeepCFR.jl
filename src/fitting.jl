function train_net_tracked!(
    net,
    x_data,
    y_data,
    w,
    batch_size,
    n_batches,
    opt;
    show_progress::Bool = true)

    isempty(x_data) && return nothing
    input_size = length(first(x_data))
    output_size = length(first(y_data))

    loss_hist = Vector{Float64}(undef, n_batches)
    X_tot = Matrix{Float32}(undef, input_size, length(x_data))
    Y_tot = Matrix{Float32}(undef, output_size, length(y_data))
    W_tot = copy(w)
    fillmat!(X_tot, x_data, 1:length(w))
    fillmat!(Y_tot, y_data, 1:length(w))

    X = Matrix{Float32}(undef, input_size, batch_size)
    Y = Matrix{Float32}(undef, output_size, batch_size)
    W = Vector{Float32}(undef, batch_size)
    sample_idxs = Vector{Int}(undef, batch_size)
    idxs = 1:length(w)

    Loss = NetLoss(net, X, Y, W)
    Loss_tot = NetLoss(net, X_tot, Y_tot, W_tot)
    p = Flux.params(net)

    prog = Progress(n_batches; enabled=show_progress)
    for i in 1:n_batches
        loss_hist[i] = Loss_tot()

        rand!(sample_idxs, idxs)
        fillmat!(X::Matrix{Float32}, x_data, sample_idxs)
        fillmat!(Y::Matrix{Float32}, y_data, sample_idxs)
        copyto!(W::Vector{Float32}, @view w[sample_idxs])

        gs = gradient(Loss, p)

        Flux.update!(opt, p::Flux.Params, gs)
        next!(prog)
    end

    return loss_hist
end

function train_net_tracked!(net,Mv::AdvantageMemory,bs,n_b,opt;show_progress::Bool = true)
    return train_net_tracked!(net, Mv.I, Mv.r, Mv.t, bs, n_b, opt; show_progress=show_progress)
end

function train_net_tracked!(net,Mπ::StrategyMemory,bs,n_b,opt;show_progress::Bool = true)
    return train_net_tracked!(net, Mπ.I, Mπ.σ, Mπ.t, bs, n_b, opt; show_progress=show_progress)
end

mutable struct LossCache
    txx::Float32
    tx::Vector{Float32}
    t::Float32
end

LossCache(r::Vector{Float32}) = LossCache(0.0f0, zero(r), 0.0f0)

"""
Lowest possible WMSE
"""
function lower_limit_loss(X::Vector{INFO},Y,W) where INFO

    d = Dict{INFO,LossCache}()

    for (k,t,r) in zip(X,W,Y)
        lc = get!(d, k) do # txxsum, txsum, tsum
            LossCache(r)
        end
        lc.txx += t*sum(abs2, r)
        lc.tx .+= t .* r
        lc.t += t
    end

    l = 0.0f0
    for lc in values(d)
        l += lc.txx - sum(abs2, lc.tx)/lc.t
    end
    return l / length(X)
end

lower_limit_loss(Mπ::StrategyMemory) = lower_limit_loss(Mπ.I, Mπ.σ, Mπ.t)
lower_limit_loss(Mv::AdvantageMemory) = lower_limit_loss(Mv.I, Mv.r, Mv.t)

function optimality_distance(net, x_data, y_data, w)
    isempty(x_data) && return 0.0
    input_size = length(first(x_data))
    output_size = length(first(y_data))

    X_tot = Matrix{Float32}(undef, input_size, length(x_data))
    Y_tot = Matrix{Float32}(undef, output_size, length(y_data))
    W_tot = copy(w)

    fillmat!(X_tot, x_data, 1:length(w))
    fillmat!(Y_tot, y_data, 1:length(w))

    l = wmse(net(X_tot), Y_tot, W_tot)
    l_min = lower_limit_loss(x_data, y_data, w)

    if iszero(l_min)
        return l
    else
        return (l - l_min) / l_min
    end
end

optimality_distance(net, Mπ::StrategyMemory)  = optimality_distance(net, Mπ.I, Mπ.σ, Mπ.t)
optimality_distance(net, Mv::AdvantageMemory) = optimality_distance(net, Mv.I, Mv.r, Mv.t)
