mutable struct DeepCFRSolver{
        GPU,
        AN,
        SN,
        INFO,
        REGRET,
        STRAT,
        G<:Game,
        ADV_OPT<:Flux.Optimise.AbstractOptimiser,
        STRAT_OPT<:Flux.Optimise.AbstractOptimiser
    }
    gpu::Val{GPU}

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
    advantage_opt::ADV_OPT
    strategy_opt::STRAT_OPT
    game::G
    T::Float32
end

on_gpu(::DeepCFRSolver{GPU}) where GPU = GPU
infotype(::DeepCFRSolver{A,S,I}) where {GPU,A,S,I} = I
regrettype(::DeepCFRSolver{A,S,I,R}) where {GPU,A,S,I,R} = R
strattype(::DeepCFRSolver{A,S,I,R,ST}) where {GPU, A,S,I,R,ST} = ST

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

`value_epochs::Int` - Number of epochs for which value/advantage/regret networks are trained

`strategy_epochs::Int` - Number of epochs for which strategy networks are trained

`buffer_size::Int` - Capacity of memory buffers (both strategy and value)

`batch_size::Int` - Batch size for each gradient update step (both strategy and value)

`traversals::Int` - Number of MCCFR tree traversals to make for regret data collection between training steps

`value_optimizer::Flux.Optimise.AbstractOptimiser` - value network optimizer

`strategy_optimizer::Flux.Optimise.AbstractOptimiser` - strategy network optimizer

`values` - Tuple of Flux.jl value networks (one network for each player)

`strategy` - Single Flux.jl strategy network

`on_gpu::Bool` - Option to push network training to GPU
"""
function DeepCFRSolver(game::Game{H,K};
    value_epochs::Int = 1,
    strategy_epochs::Int = 1,
    buffer_size::Int = 100*10^3,
    batch_size::Int = 1_000,
    traversals::Int = 100,
    value_optimizer = Flux.Optimiser(ClipValue(10.0), Descent()),
    strategy_optimizer = ADAM(),
    strategy = nothing,
    values::Union{Nothing, Tuple} = nothing,
    on_gpu::Bool = false
    ) where {H,K}

    VK = first(Base.return_types(vectorized, (typeof(game),K)))
    @assert VK <: AbstractVector "`vectorized(::Game{H,K}, ::K)` should return vector"

    if promote_type(eltype(VK), Float32) !== Float32
        @warn "Float32 eltype preferred for vectorized information state key"
    end

    value_nets = if isnothing(values)
        in_size, out_size = in_out_sizes(game)
        value_net = Chain(Dense(in_size, 10, relu), Dense(10,out_size))
        (value_net, deepcopy(value_net))
    else
        values
    end

    strategy_net = if isnothing(strategy)
        Chain(
            Dense(in_size, 20, relu),
            Dense(20, out_size, sigmoid),
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
        value_epochs,
        strategy_epochs,
        buffer_size,
        batch_size,
        traversals,
        value_optimizer,
        strategy_optimizer,
        game,
        1.0f0
    )
end

strategy(sol::DeepCFRSolver, I::AbstractVector) = sol.Π(I)

(sol::DeepCFRSolver)(I::AbstractVector) = strategy(sol, I)


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
