function checksolver(sol::DeepCFRSolver)
    game = sol.game
    h0 = initialhist(game)
    k = vectorized(game, infokey(game, h0))

    K = infokeytype(game)
    VK = first(Base.return_types(vectorized, (typeof(game),K)))
    @assert VK <: AbstractVector "vectorized game keys must be vectors"

    A = actions(game, h0)
    in_size = length(k)
    out_size = length(A)

    for (i,net) in enumerate(sol.V)
        try
            output = net(k)
        catch e
            if e isa DimensionMismatch
                throw(DimensionMismatch("Value network $i input layer size should be $(length(k))"))
            else
                rethrow(e)
            end
        end
        output = net(k)
        @assert(
            length(output) == out_size,
            "Value network $i output dim : $(length(output)) \n
            Required output dim: $out_size"
        )
    end

    # TODO: is there some `Flux.nfan` for chain?
    try
        output = sol.Π(k)
    catch e
        if e isa DimensionMismatch
            throw(DimensionMismatch("Strategy input layer size should be $(length(k))"))
        else
            rethrow(e)
        end
    end
    output = sol.Π(k)
    @assert(
        length(output) == out_size,
        "Strategy network output dim : $(length(output)) \n
        Required output dim: $out_size"
    )

    @assert(
        sum(output) ≈ 1.0,
        "Strategy network output should be normalized to 1.0"
    )

    return "Tests Pass" # better way to do this?
end
