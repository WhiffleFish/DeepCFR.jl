mutable struct Throttle{F}
    f::F
    n::Int
    state::Int
end

function Throttle(f::Function, n::Int)
    return Throttle(f, n, 0)
end

function (t::Throttle)()
    iszero(rem(t.state, t.n)) && t.f()
    t.state += 1
end

mutable struct ModelSaver{M}
    model::M
    n::Int
    path::String
    state::Int
end

function ModelSaver(model, n::Int=500; path=pwd())
    return ModelSaver(model, n, path, 0)
end

function (m::ModelSaver)()
    if iszero(rem(m.state, m.n))
        path = joinpath(m.path, "model_$(m.state).jld2")
        FileIO.save(path, Dict("model" => m.model))
    end
    m.state += 1
end

mutable struct ExploitabilityCallback{SOL<:DeepCFRSolver, ESOL<:ExploitabilitySolver}
    sol::SOL
    e_sol::ESOL
    n::Int
    state::Int
    hist::CFR.ExploitabilityHistory
end

function ExploitabilityCallback(sol::DeepCFRSolver, n::Int=1; p::Int=1)
    e_sol = ExploitabilitySolver(sol, p)
    return ExploitabilityCallback(sol, e_sol, n, 0, CFR.ExploitabilityHistory())
end

function (cb::ExploitabilityCallback)()
    if iszero(rem(cb.state, cb.n))
        sol = cb.sol
        initialize!(sol.Π)
        train_policy!(sol, sol.strategy_batches, deepcopy(sol.strategy_opt))
        e = CFR.exploitability(cb.e_sol, sol)
        push!(cb.hist, cb.state, e)
    end
    cb.state += 1
end

@recipe f(cb::ExploitabilityCallback) = cb.hist

struct FittingCallback{SOL<:AbstractDeepCFRSolver,N}
    sol::SOL
    verbose::Bool
    io::IO
    hist::NTuple{N,Vector{Float64}}
end

function FittingCallback(sol; verbose=true, io=stderr)
    return FittingCallback(sol, verbose, io, Tuple(Float64[] for _ in eachindex(sol.V)))
end

function (cb::FittingCallback)()
    sol = cb.sol
    io = cb.io
    cb.verbose && println()
    for p in eachindex(sol.Mv)
        d = optimality_distance(sol.V[p], sol.Mv[p])
        push!(cb.hist[p], d)
        if cb.verbose
            println(io, p, ": ", round(d, sigdigits=3))
        end
    end
end

@recipe f(cb::FittingCallback) = hcat(cb.hist...)
