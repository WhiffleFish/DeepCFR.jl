using CounterfactualRegret
const CFR = CounterfactualRegret
using SDA
using DeepCFR

game = SpaceGame(5,10)
sol = DeepCFRSolver(game)
@time train!(sol, 10)

@profiler train!(sol, 10)

i0 = infokey(game, initialhist(game))

sol(i0)

using StaticArrays
function DeepCFR.vectorized(game::CFR.Kuhn, I)
    p, pc, hist = I
    L = length(hist)
    h = Tuple(hist)
    leftover = ntuple(_ -> -1, 3 - L)
    SA[p, pc, h..., leftover...]::SVector{5,Int}
end

game = CFR.Kuhn()
i0 = infokey(game, initialhist(game))
@code_warntype DeepCFR.vectorized(game, i0)



##
sol = DeepCFRSolver(game)
@time train!(sol, 100)

@profiler train!(sol, 100)


##
sol = DeepCFRSolver(game)
@time train!(sol, 100)
length(sol.Mv[2])

sol.buffer_size
Base.summarysize(sol)*1e-6

@code_warntype sol(DeepCFR.vectorized(game, i0))

@test (@inferred sol(DeepCFR.vectorized(game, i0))) isa Vector
propertynames(t)

σ = @inferred sol(DeepCFR.vectorized(game, i0))
@test σ isa Vector{<:AbstractFloat}
@test sum(σ) ≈ 1.0






##
using Zygote
using Flux
@code_warntype gradient(x -> 3x^2 + 2x + 1, 5)

d = Dense(5,10)
X = rand(5, 100)
Y = rand(10, 100)

p = params(d)
@code_warntype gradient(p) do
    Flux.mse(d(X::Matrix{Float64}), Y::Matrix{Float64})::Float64
end

using FastClosures

function loss(d,X,Y)
    g = @closure gradient(p) do
        Flux.mse(d(X), Y)
    end
    return g::Zygote.Grads
end

function loss_closure(d,X,Y)
    g = @closure gradient(p) do
        Flux.mse(d(X), Y)
    end
    return g::Zygote.Grads
end

@benchmark loss(d,X,Y)

@benchmark loss_closure(d,X,Y)


##
@benchmark loss(d,X,Y)

@benchmark loss_closure(d,X,Y)
