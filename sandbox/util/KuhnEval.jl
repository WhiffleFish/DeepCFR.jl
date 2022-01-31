using CounterfactualRegret
using DeepCFR
using StaticArrays
using ProgressMeter

const CFR = CounterfactualRegret
const KUHN = CFR.Kuhn()

function DeepCFR.vectorized(game::CFR.Kuhn, I)
    p, pc, hist = I
    L = length(hist)
    h = Tuple(hist)
    leftover = ntuple(_ -> -1, 2 - L)
    SA[p, pc, h..., leftover...]::SVector{4,Int}
end

squared_error(x,y) = sum(abs2, x .- y)

const TRUTH_DICT = Dict{SVector{4,Int}, Vector{Float32}}(
    # SA[1,1,-1,-1] => [0.,0.], # DEPENDENT ON α
    SA[1,2,-1,-1] => [1.,0.],
    # SA[1,3,-1,-1] => [0.,1.], # DEPENDENT ON α

    SA[2,1,0,-1] => [2/3,1/3],
    SA[2,2,0,-1] => [1.,0.],
    SA[2,3,0,-1] => [0.,1.],

    SA[2,1,1,-1] => [1.,0.],
    SA[2,2,1,-1] => [2/3,1/3],
    SA[2,3,1,-1] => [0.,1.],

    SA[1,1,0,1] => [1.,0.],
    # SA[1,2,0,1] => [0.,1.], # DEPENDENT ON α
    SA[1,3,0,1] => [0.,1.],
)

function KuhnSSE(sol::DeepCFRSolver)
    σ_0 = sol(SA[1,2,-1,-1])
    α = σ_0[2]

    sse = 0.0
    for (I,σ) in TRUTH_DICT
        sse += squared_error(sol(I), σ)
    end

    sse += squared_error(sol(SA[1,3,-1,-1]), [1-3α,3α])
    sse += squared_error(sol(SA[1,2,0,1]), [2/3-α,1/3+α])

    return sse
end

function Kuhndebugtrain!(sol::DeepCFRSolver, N::Int; show_progress::Bool = true)
    loss_hist = Vector{Float64}(undef, N)
    DeepCFR.initialize!(sol)
    h0 = initialhist(sol.game)
    t = 0
    prog = Progress(N; enabled=show_progress)
    for i in 1:N
        for _ in 1:sol.traversals
            t += 1
            for p in 1:2
                DeepCFR.traverse(sol, h0, p, t)
            end
        end
        DeepCFR.initialize!(sol)
        DeepCFR.train_value!(sol, 1)
        DeepCFR.train_value!(sol, 2)
        DeepCFR.train_policy!(sol)
        sse = KuhnSSE(sol)
        ProgressMeter.next!(prog, showvalues=[(:SSE, sse)])
        loss_hist[i] = sse
    end
    return loss_hist
end
