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
    buffer_size::Int = 40*10^6,
    optimizer = ADAM(0.01)
    ) where {H,K}

    in_size, out_size = in_out_sizes(game)
    value_net = Chain(Dense(in_size,10), Dense(10,out_size))
    strategy_net = Chain(Dense(in_size,10), Dense(10,out_size))

    return DeepCFRSolver(
        (value_net, deepcopy(value_net)),
        (AdvantageMemory{K}(), AdvantageMemory{K}()),
        strategy_net,
        StrategyMemory{K}(),
        buffer_size,
        batch_size,
        optimizer,
        game
    )
end

struct StrategyMemory{INFO<:AbstractVector, STRAT<:AbstractVector}
    I::Vector{INFO}
    t::Vector{Int}
    σ::Vector{STRAT}
end

function StrategyMemory{INFO, STRAT}() where {INFO,STRAT}
    return StrategyMemory(INFO[], Int[], STRAT[])
end

function StrategyMemory{INFO}() where {INFO}
    return StrategyMemory(INFO[], Int[], Vector{Float64}[])
end

function Base.push!(Mπ::StrategyMemory, I, t, σ)
    push!(Mπ.I,I)
    push!(Mπ.t,t)
    push!(Mπ.σ,σ)
end

function Base.empty!(Mπ::StrategyMemory)
    empty!(Mπ.I)
    empty!(Mπ.t)
    empty!(Mπ.σ)
end

struct AdvantageMemory{INFO<:AbstractVector, REGRET<:AbstractVector}
    I::Vector{INFO}
    t::Vector{Int}
    r::Vector{REGRET}
end

function AdvantageMemory{INFO, REGRET}() where {INFO,REGRET}
    return AdvantageMemory(INFO[], Int[], REGRET[])
end

function AdvantageMemory{INFO}() where {INFO}
    return AdvantageMemory(INFO[], Int[], Vector{Float64}[])
end

function Base.push!(Mv::AdvantageMemory, I, t, r)
    push!(Mv.I,I)
    push!(Mv.t,t)
    push!(Mv.r,r)
end

function Base.empty!(Mv::AdvantageMemory)
    empty!(Mv.I)
    empty!(Mv.t)
    empty!(Mv.r)
end
