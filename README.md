# DeepCFR.jl
[![codecov](https://codecov.io/gh/WhiffleFish/DeepCFR.jl/branch/main/graph/badge.svg?token=NM2KU62FG2)](https://codecov.io/gh/WhiffleFish/DeepCFR.jl)

[Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164) (Brown et al.)


```julia
using CounterfactualRegret
import CounterfactualRegret as CFR
using StaticArrays
using DeepCFR

# Get Rock-Paper-Scissors as default CounterfactualRegret.jl matrix game
RPS = CFR.IIEMatrixGame()

#=
Information state type of matrix game is `Int`, 
so extend `vectorized` method to convert to vector 
s.t. it's able to be passed through a Flux.jl network
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



Define custom [Flux.jl](https://github.com/FluxML/Flux.jl) networks
```julia
using Flux

in_size = 1 # information state vector is of length 1 (ref `DeepCFR.vectorized`)
out_size = 3 # 3 actions: rock, paper, scissors

#= 
strategy is a probability distribution -> network output must add to 1.
Simple solution is to softmax output
=#
strategy_network = Chain(Dense(in_size, 40), Dense(40, out_size), softmax)

# regret/value does not need to be normalized
value_network = Chain(Dense(in_size, 20), Dense(20, out_size))

sol = DeepCFRSolver(
        RPS; 
        strategy = strategy_network,
        values = (value_network, deepcopy(value_network)) 
) # DeepCFR requires as many value networks as there are players (2 here)
```