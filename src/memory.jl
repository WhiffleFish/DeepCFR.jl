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
