struct OutcomeSample
    ϵ::Float32
    OutcomeSample(x=0.60) = new(x)
end

function (outcome_traverse::OutcomeSample)(sol::DeepCFRSolver, h, p, t, π_i=1.0f0, π_ni=1.0f0, s=1.0f0)
    game = sol.game
    current_player = player(game, h)
    ϵ = outcome_traverse.ϵ

    if isterminal(game, h)
        return Float32(utility(game, p, h)/s) , 1.0f0

    elseif iszero(current_player)
        a = chance_action(game, h)
        h′ = next_hist(game, h, a)
        return outcome_traverse(sol, h′, p, t, π_i, π_ni, s)

    elseif current_player == p
        I = vectorized(game, infokey(game, h)) # info KEY here, not infostate
        A = actions(game, h)
        σ = regret_match_strategy(sol, I, p)

        σ′ = ϵ*inv(length(A)) .+ (1-ϵ) .* σ # TODO: can probably get around allocation here
        a_idx = weighted_sample(σ′)
        a = A[a_idx]
        h′ = next_hist(game, h, a)
        u, π_tail = outcome_traverse(sol, h′, p, t, π_i*σ[a_idx], π_ni, s*σ′[a_idx])

        W = u*π_ni
        r̃ = Vector{Float32}(undef, length(A))
        for (k, a′) in enumerate(A)
            r̃[k] = if k == a_idx
                W*π_tail*(1 - σ[a_idx])
            else
                -W*σ[a_idx]
            end
        end

        push!(sol.Mv[p], I,t,r̃)

        return u, π_tail*σ[a_idx]
    else
        I = vectorized(game, infokey(game, h)) # info KEY here, not infostate
        A = actions(game, h)
        σ = regret_match_strategy(sol, I, current_player)

        a_idx = weighted_sample(σ)
        a = A[a_idx]
        h′ = next_hist(game, h, a)
        u, π_tail = outcome_traverse(sol, h′, p, t, π_i, π_ni*σ[a_idx], s*σ[a_idx])

        push!(sol.Mπ, I,t,σ)
        return u, π_tail*σ[a_idx]
    end
end
