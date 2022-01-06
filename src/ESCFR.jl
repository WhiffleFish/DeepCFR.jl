"""
Not confident in this MCCFR traversal implementation demonstrated in https://arxiv.org/pdf/1811.00164.pdf

Utility is not being weighted by path probability
"""
function traverse(sol::DeepCFRSolver, h, p, t)
    game = sol.game

    if isterminal(game, h)
        return utility(game, p, h)

    elseif player(game, h) === 0
        a = chance_action(game, h)
        h′ = next_hist(game,h,a)
        return traverse(game, h′, p, t)

    elseif player(game, h) == p
        I = infokey(game, h) # info KEY here, not infostate
        A = actions(game, h)
        v_σ_Ia = Vector{Float64}(undef, length(A))
        v_σ = 0.0

        σ = regret_match_strategy(sol, I, p)
        for (i,a) in enumerate(A)
            h′ = next_hist(game, h, a)
            v = traverse(game, h′, p, t)
            v_σ_Ia[i] = v
            v_σ += σ[i]*v
        end

        # TODO: Ensure there isn't some positivity restriction on regret
        r = v_σ_Ia .-= v_σ
        push!(sol.Mv, I,t,r)

    else
        I = infokey(game, h) # info KEY here, not infostate
        A = actions(game, h)
        σ = regret_match_strategy(sol, I, other_player(p))
        push!(sol.Mπ, I,t,σ)
        a = A[weighted_sample(σ)]
        h′ = next_hist(game, h, a)
        return traverse(game, h′, p, t)
    end
end

"""
Sample index of vector according to weights given by vector (sum of weights assumed to be 1.0)
"""
function weighted_sample(rng::AbstractRNG, σ::AbstractVector{Float64})
    t = rand(rng)
    i = 1
    cw = σ[1]
    while cw < t && i < length(σ)
        i += 1
        @inbounds cw += σ[i]
    end
    return i
end

weighted_sample(σ::AbstractVector{Float64}) = weighted_sample(Random.GLOBAL_RNG, σ)

function regret_match_strategy(sol::DeepCFRSolver, I, p)
    # TODO: ensure that mutating this array isn't mutating something else
    values = sol.V[p](I)
    s = 0.0f0
    @inbounds for i in eachindex(values)
        if values[i] > 0.0f0
            s += values[i]
        else
            values[i] = 0.f0
        end
    end
    
    if s > 0.0f0
        return values ./= s
    else
        return fill!(values, inv(length(values)))
    end
end
