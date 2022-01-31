using CounterfactualRegret
using Flux
using DeepCFR
using ProgressMeter

const CFR = CounterfactualRegret
const KUHN = CFR.Kuhn()

sol = DeepCFRSolver(
    KUHN;
    strategy_epochs = 5,
    value_epochs = 5,
    batch_size = 10_000,
    optimizer = Flux.Optimiser(ClipValue(10.0), Descent()))
lh1, lh2, lh3 = value_consistency!(sol, 500)

plot(
    0:5,
    lh1,
    label="",
    c=:blue,
    alpha=0.1,
    yscale=:log10,
    title = "Gradient Clipped SGD (10.0)",
    xlabel = "Training Epoch",
    ylabel = "Advantage Network MSE")
title!("Gradient Clipped SGD (1.0)")
xlabel!("Training Epoch")
ylabel!("Advantage Network MSE")
savefig()



##
sol = DeepCFRSolver(
    KUHN;
    strategy_epochs = 10,
    value_epochs = 5,
    batch_size = 10_000)
lh1, lh2, strat_loss_hist = value_consistency!(sol, 500)

plot(
    0:10,
    strat_loss_hist,
    label = "",
    c = :blue,
    alpha = 0.2,
    # yscale = :log10,
    title = "ADAM(0.01)",
    xlabel = "Training Epoch",
    ylabel = "Strategy Network MSE")
