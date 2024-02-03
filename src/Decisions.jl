"Functions for expected utility calculations"
module Decisions

using NamedArrays
using IterTools

include("observations.jl")
include("losses.jl")


export ObservationMatrix
export LossMatrix
export expected_strategy_loss
export bayes_strategy_loss

end
