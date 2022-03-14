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
