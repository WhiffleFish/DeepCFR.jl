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

    value_epochs::Int
    strategy_epochs::Int
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
    k = vectorized(game, infokey(game, h0))
    A = actions(game, h0)
    in_size = length(k)
    out_size = length(A)
    return in_size, out_size
end

function DeepCFRSolver(game::Game{H,K};
    value_epochs::Int = 1,
    strategy_epochs::Int = 1,

    buffer_size::Int = 40*10^3,
    batch_size::Int = 40,
    traversals::Int = 40,
    optimizer = ADAM(0.01)
    ) where {H,K}

    VK = first(Base.return_types(vectorized, (typeof(game),K)))
    @assert VK <: AbstractVector
    in_size, out_size = in_out_sizes(game)
    value_net = Chain(Dense(in_size,10), Dense(10,out_size))
    strategy_net = Chain(Dense(in_size,10), Dense(10,out_size), softmax)

    return DeepCFRSolver(
        (value_net, deepcopy(value_net)),
        (AdvantageMemory{VK}(buffer_size), AdvantageMemory{VK}(buffer_size)),
        strategy_net,
        StrategyMemory{VK}(buffer_size),
        value_epochs,
        strategy_epochs,
        buffer_size,
        batch_size,
        traversals,
        optimizer,
        game
    )
end

function strategy(sol::DeepCFRSolver, I::AbstractVector)
    σ = sol.Π(I)
    return σ ./= sum(σ)
end

(sol::DeepCFRSolver)(I::AbstractVector) = strategy(sol, I)


function Base.show(io::IO, mime::MIME"text/plain", sol::DeepCFRSolver)
    Lv = length(first(sol.Mv))
    Lπ = length(sol.Mπ)
    sz = sol.buffer_size
    println(io, "\n\t Deep CFR Solver")
    println(io, join(fill('_', 20)))
    println(io, "Advantage Net    | \t $(string(first(sol.V)))")
    println(io, "Advantage Memory | \t $Lv / $sz")
    println(io, "Strategy Net     | \t $(string(sol.Π))")
    println(io, "Strategy Memory  | \t $Lπ / $sz")
    println(io, "Batch Size       | \t $(sol.batch_size)")
    println(io, "Traversals       | \t $(sol.traversals)")
    println(io, "Optimizer        | \t $(typeof(sol.optimizer))")
    println(io, "Game             | \t $(typeof(sol.game))")

    nothing
end
