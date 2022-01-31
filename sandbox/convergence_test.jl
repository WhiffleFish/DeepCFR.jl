include(joinpath(@__DIR__, "util", "KuhnEval.jl"))

sol = DeepCFRSolver(KUHN; strategy_epochs=10, value_epochs=5, batch_size=10_000)
loss_hist = Kuhndebugtrain!(sol, 500)

using Plots
plot(
    loss_hist,
    label="",
    title = "Deep CFR training progress (GC SGD)",
    xlabel = "Training Iteration",
    ylabel = "SSE Strategy loss"
    )

savefig("DeepCFRTrainingProgressGCSGD.svg")

Base.floor(T::Type) = Base.Fix1(floor, T)
f = floor(Int)
f(3.5)

DeepCFR.train_policy!(sol, 20)

KuhnSSE(sol)
KuhnSSE(sol)
