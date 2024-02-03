"Functions for expected utility calculations"
module Decisions

using NamedArrays
using IterTools

include("losses.jl")
include("observations.jl")

export LossMatrix
export ObservationMatrix

export expected_strategy_loss
export bayes_strategy_loss

end
