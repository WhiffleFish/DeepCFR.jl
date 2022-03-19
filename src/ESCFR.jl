function traverse(sol::DeepCFRSolver, h, p, t)
    game = sol.game

    if isterminal(game, h)
        return Float32(CounterfactualRegret.utility(game, p, h))

    elseif player(game, h) == 0
        a = chance_action(game, h)
        h′ = next_hist(game,h,a)
        return traverse(sol, h′, p, t)

    elseif player(game, h) == p
        I = vectorized(game, infokey(game, h)) # info KEY here, not infostate
        A = actions(game, h)
        v_σ_Ia = Vector{Float32}(undef, length(A))
        v_σ = 0.0f0

        σ = regret_match_strategy(sol, I, p)
        for (i,a) in enumerate(A)
            h′ = next_hist(game, h, a)
            v = traverse(sol, h′, p, t)
            v_σ_Ia[i] = v
            v_σ += σ[i]*v
        end

        r = v_σ_Ia .-= v_σ
        push!(sol.Mv[p], I,t,r)

        return v_σ

    else
        I = vectorized(game, infokey(game, h)) # info KEY here, not infostate
        A = actions(game, h)
        σ = regret_match_strategy(sol, I, other_player(p))
        push!(sol.Mπ, I,t,σ)
        a = A[weighted_sample(σ)]
        h′ = next_hist(game, h, a)
        return traverse(sol, h′, p, t)
    end
end

"""
Infokey type of game may not be in vectorized form.

`vectorized(game::Game, I::infokeytype(game))` returns the original key type
in vectorized form to be pushed through a neural network.
"""
function vectorized end

vectorized(game::Game, I) = I

"""
Sample index of vector according to weights given by vector (sum of weights assumed to be 1.0)
"""
function weighted_sample(rng::Random.AbstractRNG, σ::AbstractVector)
    t = rand(rng)
    i = 1
    cw = σ[1]
    while cw < t && i < length(σ)
        i += 1
        cw += σ[i]
    end
    return i
end

weighted_sample(σ::AbstractVector) = weighted_sample(Random.GLOBAL_RNG, σ)

function regret_match_strategy(sol::DeepCFRSolver, I::AbstractVector, p)
    return regret_match_strategy(sol.V[p](I))
end

function regret_match_strategy(r::AbstractVector)
    s = 0.0f0
    for i in eachindex(r)
        if r[i] > 0.0f0
            s += r[i]
        else
            v = r[i]
            r[i] = 0.0f0
        end
    end

    if s > 0.0f0
        return r ./= s
    else
        return fill!(r, inv(length(r)))
    end
end


# Using regret matching strategy from paper doesn't seem to work all that well
#=
function regret_match_strategy(r::AbstractVector)
    s = 0.0f0
    max_neg_idx = 0
    max_neg = -Inf
    for i in eachindex(r)
        if r[i] > 0.0f0
            s += r[i]
        else
            v = r[i]
            if v > max_neg
                max_neg_idx = i
                max_neg = v
            end
            r[i] = 0.0f0
        end
    end

    if s > 0.0f0
        return r ./= s
    else
        r[max_neg_idx] = 1.0f0
        return r
    end
end
=#
