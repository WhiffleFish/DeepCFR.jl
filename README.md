# DeepCFR.jl

[Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164) (Brown et al.)


```julia
using CounterfactualRegret
import CounterfactualRegret as CFR
using StaticArrays
using DeepCFR

# Get Rock-Paper-Scissors as default CounterfactualRegret.jl matrix game
RPS = CFR.IIEMatrixGame()

#=
Information state type of matrix game is `Int`, so extend `vectorized` method to convert to vector s.t. it's able to be passed through a Flux.jl network
=#
DeepCFR.vectorized(::CFR.IIEMatrixGame, I) = SA[Float32(I)]


sol = DeepCFRSolver(
        RPS; 
        buffer_size = 100*10^3, 
        batch_size = 128, 
        traversals = 10, 
        on_gpu = false
)

# train CFR solver for 1000 iterations
train!(sol, 1_000, show_progress=true)

I0 = DeepCFR.vectorized(0) # information state corresponding to first player's turn
I1 = DeepCFR.vectorized(1) # information state corresponding to second player's turn

sol(I0) # return strategy for player 1 
sol(I1) # return strategy for player 2
```