struct NullInfoState <: CounterfactualRegret.AbstractInfoState end
abstract type AbstractDeepCFRSolver{G} <: CFR.AbstractCFRSolver{Nothing,G,NullInfoState} end
mutable struct DeepCFRSolver{
        GPU,
        AN,
        SN,
        INFO,
        REGRET,
        STRAT,
        G <: Game,
        ADV_OPT <: Flux.Optimise.AbstractOptimiser,
        STRAT_OPT <: Flux.Optimise.AbstractOptimiser
    } <: AbstractDeepCFRSolver{G}
    gpu::Val{GPU}

    "Advantage networks"
    V::NTuple{2,AN}

    "Advantage memory"
    Mv::NTuple{2,AdvantageMemory{INFO, REGRET}}

    "Strategy Network"
    Π::SN

    "Strategy Memory"
    Mπ::StrategyMemory{INFO, STRAT}

    value_batches::Int
    strategy_batches::Int
    buffer_size::Int
    batch_size::Int
    traversals::Int
    advantage_opt::ADV_OPT
    strategy_opt::STRAT_OPT
    game::G
    T::Float32
end

on_gpu(::DeepCFRSolver{GPU}) where GPU = GPU
infotype(::DeepCFRSolver{GPU,A,S,I}) where {GPU,A,S,I} = I
regrettype(::DeepCFRSolver{GPU,A,S,I,R}) where {GPU,A,S,I,R} = R
strattype(::DeepCFRSolver{GPU,A,S,I,R,ST}) where {GPU,A,S,I,R,ST} = ST

function in_out_sizes(game::Game)
    h0 = initialhist(game)
    k = vectorized(game, infokey(game, h0))
    A = actions(game, h0)
    in_size = length(k)
    out_size = length(A)
    return in_size, out_size
end

"""
`game::CounterfactualRegret.Game`

`value_batches::Int` - Number of epochs for which value/advantage/regret networks are trained

`strategy_batches::Int` - Number of epochs for which strategy networks are trained

`buffer_size::Int` - Capacity of memory buffers (both strategy and value)

`batch_size::Int = ` - Batch size for each gradient update step (both strategy and value)

`traversals::Int = 100` - Number of MCCFR tree traversals to make for regret data collection between training steps

`value_optimizer::Flux.Optimise.AbstractOptimiser` - value network optimizer

`strategy_optimizer::Flux.Optimise.AbstractOptimiser` - strategy network optimizer

`values` - Tuple of Flux.jl value networks (one network for each player)

`strategy` - Single Flux.jl strategy network

`on_gpu::Bool = false` - Option to push network training to GPU
"""
function DeepCFRSolver(game::Game{H,K};
    value_batches::Int = 100,
    strategy_batches::Int = 1000,
    buffer_size::Int = 100*10^3,
    batch_size::Int = 512,
    traversals::Int = 50,
    value_optimizer = ADAM(5e-3),
    strategy_optimizer = ADAM(5e-3),
    strategy = nothing,
    values::Union{Nothing, Tuple} = nothing,
    on_gpu::Bool = false
    ) where {H,K}

    VK = first(Base.return_types(vectorized, (typeof(game),K)))
    @assert VK <: AbstractVector "`vectorized(::Game{H,K}, ::K)` should return vector"

    #= Leave to checks
    if promote_type(eltype(VK), Float32) !== Float32
        @warn "Float32 eltype preferred for vectorized information state key"
    end
    =#

    in_size, out_size = in_out_sizes(game)
    value_nets = if isnothing(values)
        value_net = Chain(Dense(in_size, 16, sigmoid), Dense(16,out_size))
        (value_net, deepcopy(value_net))
    else
        values
    end

    strategy_net = if isnothing(strategy)
        Chain(
            Dense(in_size, 16, sigmoid),
            Dense(16, 16, sigmoid),
            Dense(16, out_size, sigmoid),
            softmax)
    else
        strategy
    end

    return DeepCFRSolver(
        Val(on_gpu),
        value_nets,
        (AdvantageMemory{VK}(buffer_size), AdvantageMemory{VK}(buffer_size)),
        strategy_net,
        StrategyMemory{VK}(buffer_size),
        value_batches,
        strategy_batches,
        buffer_size,
        batch_size,
        traversals,
        value_optimizer,
        strategy_optimizer,
        game,
        1.0f0
    )
end

function CFR.strategy(sol::DeepCFRSolver, I)
    game = sol.game
    if typeof(I) === infokeytype(game)
        return sol.Π(vectorized(game, I))
    else
        return sol.Π(I)
    end
end

(sol::DeepCFRSolver)(I) = CFR.strategy(sol, I)


function Base.show(io::IO, mime::MIME"text/plain", sol::DeepCFRSolver)
    Lv = length(first(sol.Mv))
    Lπ = length(sol.Mπ)
    sz = sol.buffer_size
    println(io, "\n\t Deep CFR Solver")
    println(io, join(fill('_', 20)))
    println(io, "GPU                 | \t $(on_gpu(sol))")
    println(io, "Advantage Net       | \t $(string(first(sol.V)))")
    println(io, "Advantage Memory    | \t $Lv / $sz")
    println(io, "Strategy Net        | \t $(string(sol.Π))")
    println(io, "Strategy Memory     | \t $Lπ / $sz")
    println(io, "Batch Size          | \t $(sol.batch_size)")
    println(io, "Traversals          | \t $(sol.traversals)")
    println(io, "Advantage Optimizer | \t $(typeof(sol.advantage_opt))")
    println(io, "Strategy Optimizer  | \t $(typeof(sol.strategy_opt))")
    println(io, "Game                | \t $(typeof(sol.game))")

    nothing
end
