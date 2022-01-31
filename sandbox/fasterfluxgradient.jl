using Flux
using Zygote

d = Dense(3,2)

X = [1,2,3]
Y = [3,1]

using Cthulhu
using FastClosures
p = params(d)
@descend @closure gradient(p) do
    Flux.mse(d(X),Y)
end


function lossgrad(net, X, Y)
    return @closure gradient(p) do
        Flux.mse(net(X),Y; agg = x -> mean(x))
    end
end



lossgrad(d,X,Y)
using JET
JET.@report_opt lossgrad(d,X,Y)

JET.@report_opt lossgrad(d,X,Y)
