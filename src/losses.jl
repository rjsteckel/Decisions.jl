

struct LossMatrix
    loss::NamedArray
    actions::Array{String}
    states::Array{String}

    LossMatrix(losses::Array, actions::Array{String}, states::Array{String}) = begin
        checks = (
            size(losses)[1] == length(actions) &&
            size(losses)[2] == length(states)
        )
        @assert checks "dimension check failed" 

        losses = NamedArray(
            [0 5; 1 3; 3 2],
            names=(actions, states),
            dimnames=("actions", "states")
        )
        new(losses, actions, states)
    end
end



struct StrategyLossMatrix
    L::LossMatrix
    P::Vector{Tuple}
    A::Dict
    SL::Vector

    StrategyLossMatrix(L::LossMatrix) = begin
        # all permutations of actions/states (possible strategies)        
        P = IterTools.product(fill(L.actions, length(L.actions)) ...) |> collect
        P = vec(P)
        for i in 1:length(P)
            P[i] = reverse(P[i])  # to match book
        end

        # action index
        A = Dict(zip(L.actions, 1:length(L.actions)))
        
        # Compute losses for each strategy
        strategy_loss = Vector()
        for i in 1:length(P)
            push!(strategy_loss, vcat([L.loss[[i],:] for i in [A[a] for a in P[i]]]...))
        end
        new(L, P, A, strategy_loss)
    end
end