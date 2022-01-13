# TODO: Do we really need to parameterize REGRET, STRAT or should we just leave as Vector{Float64} ?
struct DeepCFRSolver{AN, SN, INFO, REGRET, STRAT, G<:Game, OPT<:Flux.Optimise.AbstractOptimiser}
    "Advantage networks"
    V::NTuple{2,AN}

    "Advantage memory"
    Mv::NTuple{2,AdvantageMemory{INFO, REGRET}}

    "Strategy Network"
    Π::SN

    "Strategy Memory"
    Mπ::StrategyMemory{INFO, STRAT}

    buffer_size::Int
    batch_size::Int
    traversals::Int
    optimizer::OPT
    game::G
end

infotype(::DeepCFRSolver{A,S,I}) where {A,S,I} = I
regrettype(::DeepCFRSolver{A,S,I,R}) where {A,S,I,R} = R
strattype(::DeepCFRSolver{A,S,I,R,ST}) where {A,S,I,R,ST} = ST

function in_out_sizes(game::Game)
    h0 = initialhist(game)
    k = infokey(game, h0)
    A = actions(game, h0)
    in_size = length(k)
    out_size = length(A)
    return in_size, out_size
end

function DeepCFRSolver(game::Game{H,K};
    buffer_size::Int = 40*10^3,
    batch_size::Int = 40,
    traversals::Int = 40,
    optimizer = ADAM(0.01)
    ) where {H,K}

    in_size, out_size = in_out_sizes(game)
    value_net = Chain(Dense(in_size,10), Dense(10,out_size))
    strategy_net = Chain(Dense(in_size,10), Dense(10,out_size))

    return DeepCFRSolver(
        (value_net, deepcopy(value_net)),
        (AdvantageMemory{K}(buffer_size), AdvantageMemory{K}(buffer_size)),
        strategy_net,
        StrategyMemory{K}(buffer_size),
        buffer_size,
        batch_size,
        traversals,
        optimizer,
        game
    )
end

function strategy(sol::DeepCFRSolver, I)
    σ = sol.Π(I)
    return σ ./= sum(σ)
end
