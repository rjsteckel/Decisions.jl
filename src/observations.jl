

struct ObservationMatrix
    O::NamedArray

    ObservationMatrix(probs::Array, states::Array{String}, obs::Array{String}) = begin
        checks = (
            size(probs)[1] == length(states) &&
            size(probs)[2] == length(obs)
        )
        @assert checks "dimension check failed" 
        O = NamedArray(
            probs,
            names=(states, obs),
            dimnames=("states", "obs")
        )
        new(O)
    end
end