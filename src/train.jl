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
        train_value!(sol, sol.batch_size, 1)
        train_value!(sol, sol.batch_size, 2)
    end
    train_policy!(sol, sol.batch_size)
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

function train_value!(sol::DeepCFRSolver, batch_size::Int, p::Int)
    Mv = sol.Mv[p]
    V = sol.V[p]
    opt = sol.optimizer

    L = length(Mv.t)
    iszero(L) && return
    full_batches, leftover = divrem(L, batch_size)
    total_batches = full_batches
    !iszero(leftover) && (total_batches += 1)

    input_size = length(first(Mv.I))
    output_size = length(first(Mv.r))
    perm = randperm(L)
    perms = Flux.chunk(perm, total_batches)

    I = Matrix{eltype(infotype(sol))}(undef, input_size, batch_size)
    R = Matrix{Float64}(undef, output_size, batch_size)

    # handle full batches
    for i in 1:full_batches
        fillmat!(I, Mv.I[perms[i]])
        fillmat!(R, Mv.r[perms[i]])
        p = params(V)
        gs = gradient(p) do
            Flux.Losses.mse(V(I), R)
        end
        Flux.update!(opt, p, gs)
    end

    # whatever is leftover
    if !iszero(leftover)
        I = Matrix{eltype(infotype(sol))}(undef, input_size, leftover)
        R = Matrix{Float64}(undef, output_size, leftover)

        fillmat!(I, Mv.I[last(perms)])
        fillmat!(R, Mv.r[last(perms)])
        p = params(V)
        gs = gradient(p) do
            Flux.Losses.mse(V(I),R)
        end
        Flux.update!(opt, p, gs)
    end
    nothing
end


function train_policy!(sol::DeepCFRSolver, batch_size::Int)
    Mπ = sol.Mπ
    Π = sol.Π
    opt = sol.optimizer

    L = length(Mπ.t)
    iszero(L) && return
    full_batches, leftover = divrem(L, batch_size)
    total_batches = full_batches
    !iszero(leftover) && (total_batches += 1)

    input_size = length(first(Mπ.I))
    output_size = length(first(Mπ.σ))
    perm = randperm(L)
    perms = Flux.chunk(perm, total_batches)

    I = Matrix{eltype(infotype(sol))}(undef, input_size, batch_size)
    σ = Matrix{Float64}(undef, output_size, batch_size)

    # TODO: DRY
    # handle full batches
    for i in 1:full_batches
        fillmat!(I, Mπ.I[perms[i]])
        fillmat!(σ, Mπ.σ[perms[i]])
        p = params(Π)
        gs = gradient(p) do
            Flux.Losses.mse(Π(I), σ)
        end
        Flux.update!(opt, p, gs)
    end

    # whatever is leftover
    if !iszero(leftover)
        I = Matrix{eltype(infotype(sol))}(undef, input_size, leftover)
        σ = Matrix{Float64}(undef, output_size, leftover)

        fillmat!(I, Mπ.I[last(perms)])
        fillmat!(σ, Mπ.σ[last(perms)])
        p = params(Π)
        gs = gradient(p) do
            Flux.Losses.mse(Π(I),σ)
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
