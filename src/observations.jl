
"""
    ObservationMatrix(probs, states, obs)

A structure to store the probability of observations given states: P(x|θ).
Uses `NamedArrays` to allow indexing by state and observation names.
"""
struct ObservationMatrix
    states_observations::NamedArray{Float64, 2}

    function ObservationMatrix(probs::AbstractMatrix{<:Real}, states::AbstractVector{<:AbstractString}, obs::AbstractVector{<:AbstractString})
        @assert size(probs) == (length(states), length(obs)) "Dimension mismatch: size(probs) must be (length(states), length(obs))" 
        O = NamedArray(
            Float64.(probs),
            names=(String.(states), String.(obs)),
            dimnames=("states", "obs")
        )
        new(O)
    end
end
