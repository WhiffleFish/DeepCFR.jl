function checksolver(sol::DeepCFRSolver)
    h0 = initialhist(game)
    k = vectorized(game, infokey(game, h0))

    VK = first(Base.return_types(vectorized, (typeof(game),K)))
    @assert VK <: AbstractVector "vectorized game keys must be vectors"

    A = actions(game, h0)
    in_size = length(k)
    out_size = length(A)

    for (i,net) in enumerate(sol.V)
        output = net(k)
        @assert(
            length(output)==out_size,
            "Value network $i output dim : $(length(output)) \n
            Required output dim: $out_size"
        )
    end

    output = Π(k)
    @assert(
        length(output) == out_size,
        "Strategy network output dim : $(length(output)) \n
        Required output dim: $out_size"
    )

    @assert(
        sum(output) ≈ 1.0,
        "Strategy network output should be normalized to 1.0"
    )
    nothing
end
