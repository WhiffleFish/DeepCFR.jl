struct StrategyMemory{INFO<:AbstractVector, STRAT<:AbstractVector}
    I::Vector{INFO}
    t::Vector{Int}
    σ::Vector{STRAT}
    capacity::Int
end

function StrategyMemory{INFO, STRAT}(sz::Int) where {INFO,STRAT}
    return StrategyMemory(INFO[], Int[], STRAT[], sz)
end

function StrategyMemory{INFO}(sz::Int) where {INFO}
    return StrategyMemory(INFO[], Int[], Vector{Float64}[], sz)
end

function Base.push!(Mπ::StrategyMemory, I, t, σ)
    if length(Mπ.t) < Mπ.capacity
        push!(Mπ.I,I)
        push!(Mπ.t,t)
        push!(Mπ.σ,σ)
    else
        popfirst!(Mπ.I)
        push!(Mπ.I,I)
        popfirst!(Mπ.t)
        push!(Mπ.t,t)
        popfirst!(Mπ.σ)
        push!(Mπ.σ,σ)
    end
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
    capacity::Int
end

function Base.length(mem::AdvantageMemory)
    return length(mem.t)
end

function Base.length(mem::StrategyMemory)
    return length(mem.t)
end

function AdvantageMemory{INFO, REGRET}(sz::Int) where {INFO,REGRET}
    return AdvantageMemory(INFO[], Int[], REGRET[], sz)
end

function AdvantageMemory{INFO}(sz::Int) where {INFO}
    return AdvantageMemory(INFO[], Int[], Vector{Float64}[], sz)
end

function Base.push!(Mv::AdvantageMemory, I, t, r)
    if length(Mv.t) < Mv.capacity
        push!(Mv.I,I)
        push!(Mv.t,t)
        push!(Mv.r,r)
    else
        popfirst!(Mv.I)
        push!(Mv.I,I)
        popfirst!(Mv.t)
        push!(Mv.t,t)
        popfirst!(Mv.r)
        push!(Mv.r,r)
    end
end

function Base.empty!(Mv::AdvantageMemory)
    empty!(Mv.I)
    empty!(Mv.t)
    empty!(Mv.r)
end
