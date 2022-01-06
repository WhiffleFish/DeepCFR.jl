#=
https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/cfr/deep_cfr.jl
=#
function CounterfactualRegret.train!(sol::DeepCFRSolver, N::Int)
    initialize!(sol)
    h0 = initialstate(sol.game)
    for t in 1:N
        for p in 1:2
            traverse(sol, h0, p, t)
        end
        train_value!(sol)
    end
    train_strategy!(sol)
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

# ```julia
#   for d in training_set # assuming d looks like (data, labels)
#     # our super logic
#     gs = gradient(params(m)) do #m is our model
#       l = loss(d...)
#     end
#     update!(opt, params(m), gs)
#   end
# ```


function train_value!(sol::DeepCFRSolver, batch_size::Int, p::Int)
    Mv = sol.Mv[p]
    V = sol.V[p]
    opt = sol.optimizer


    #= TODO: does not have to be the case that memory is evenly divisible by batch size
    But we simplify it to be such for a minimal working example
    =#
    L = length(Mv.t)
    n_batches = L ÷ batch_size
    perm = randperm(L)
    perms = Flux.chunk(perm, n_batches)

    I = Matrix{eltype(infotype(sol))}(undef, input_size, n_batches)
    R = Matrix{Float64}(undef, input_size, n_batches)

    for i in 1:n_batches
        fillmat!(I, Mv.I[perms[i]])
        fillmat!(r, Mv.r[perms[i]])
        p = params(V)
        gs = gradient(p) do
            Flux.Losses.mse(V(I),r)
        end
        update!(opt, p, gs)
    end
end

function train_policy!(sol::DeepCFRSolver, batch_size::Int)
    Mπ = sol.Mπ
    Π = sol.Π
    opt = sol.optimizer


    #= TODO: does not have to be the case that memory is evenly divisible by batch size
    But we simplify it to be such for a minimal working example
    =#
    L = length(Mπ.t)
    n_batches = L ÷ batch_size
    perm = randperm(L)
    perms = Flux.chunk(perm, n_batches)

    I = Matrix{eltype(infotype(sol))}(undef, input_size, n_batches)
    σ = Matrix{Float64}(undef, input_size, n_batches)

    for i in 1:n_batches
        fillmat!(I, Mπ.I[perms[i]])
        fillmat!(σ, Mπ.r[perms[i]])
        p = params(Π)
        gs = gradient(p) do
            Flux.Losses.mse(Π(I),σ)
        end
        update!(opt, p, gs)
    end
end

function fillmat!(mat::Matrix{T}, vecvec::Vector{Vector{T}}) where T
    inner_sz, outer_sz = size(mat)
    @assert length(vecvec) == outer_sz
    @assert length(first(vecvec)) == inner_sz
    for i in eachindex(vecvec)
        mat[:,i] .= vecvec[i]
    end
    mat
end
