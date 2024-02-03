

struct LossMatrix
    loss::NamedArray    
    states::Array{String}
    actions::Array{String}

    LossMatrix(losses::Array, states::Array{String}, actions::Array{String}) = begin
        checks = (
            size(losses)[1] == length(states) &&
            size(losses)[2] == length(actions)
        )
        @assert checks "dimension check failed" 

        losses = NamedArray(
            losses,
            names=(states, actions),
            dimnames=("states", "actions")
        )
        new(losses, states, actions)
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
        for sᵢ in eachindex(P)
            push!(strategy_loss, [A[s] for s in P[sᵢ]])
            #push!(strategy_loss, vcat([L.loss[:,[i]] for i in [A[a] for a in P[s]]]...))
        end
        new(L, P, A, strategy_loss)
    end
end


function Base.show(io::IO, ::MIME"text/plain", l::LossMatrix)
    println(io, l.loss)
end


