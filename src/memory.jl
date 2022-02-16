mutable struct StrategyMemory{INFO<:AbstractVector, STRAT<:AbstractVector}
    I::Vector{INFO}
    t::Vector{Int}
    σ::Vector{STRAT}
    capacity::Int
    i::Int
end

function StrategyMemory{INFO, STRAT}(sz::Int) where {INFO,STRAT}
    return StrategyMemory(
        sizehint!(INFO[], sz),
        sizehint!(Int[], sz),
        sizehint!(STRAT[], sz),
        sz,
        0
    )
end

function StrategyMemory{INFO}(sz::Int) where {INFO}
    return StrategyMemory{INFO, Vector{Float32}}(sz)
end

function Base.push!(Mπ::StrategyMemory, I, t, σ)
    i = (Mπ.i += 1)
    k = Mπ.capacity
    if i ≤ k
        push!(Mπ.I,I)
        push!(Mπ.t,t)
        push!(Mπ.σ,σ)
    else
        j = rand(1:i)
        if j ≤ k
            Mπ.I[j] = I
            Mπ.t[j] = t
            Mπ.σ[j] = σ
        end
    end
end

function Base.empty!(Mπ::StrategyMemory)
    Mπ.i = 0
    empty!(Mπ.I)
    empty!(Mπ.t)
    empty!(Mπ.σ)
end

mutable struct AdvantageMemory{INFO<:AbstractVector, REGRET<:AbstractVector}
    I::Vector{INFO}
    t::Vector{Int}
    r::Vector{REGRET}
    capacity::Int
    i::Int
end

function Base.length(mem::AdvantageMemory)
    return length(mem.t)
end

function Base.length(mem::StrategyMemory)
    return length(mem.t)
end

function AdvantageMemory{INFO, REGRET}(sz::Int) where {INFO,REGRET}
    return AdvantageMemory(
        sizehint!(INFO[], sz),
        sizehint!(Int[], sz),
        sizehint!(REGRET[], sz),
        sz,
        0
    )
end

function AdvantageMemory{INFO}(sz::Int) where {INFO}
    return AdvantageMemory{INFO, Vector{Float32}}(sz)
end

function Base.push!(Mv::AdvantageMemory, I, t, r)
    i = (Mv.i += 1)
    k = Mv.capacity
    if i ≤ k
        push!(Mv.I,I)
        push!(Mv.t,t)
        push!(Mv.r,r)
    else
        j = rand(1:i)
        if j ≤ k
            Mv.I[j] = I
            Mv.t[j] = t
            Mv.r[j] = r
        end
    end
end

function Base.empty!(Mv::AdvantageMemory)
    Mv.i = 0
    empty!(Mv.I)
    empty!(Mv.t)
    empty!(Mv.r)
end
