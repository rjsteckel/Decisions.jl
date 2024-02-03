"Functions for expected utility calculations"
module Decisions

using NamedArrays
using IterTools

include("losses.jl")
include("observations.jl")
include("strategies.jl")


export LossMatrix
"""
    Loss Matrix
    L = LossMatrix([2 3; 1 1], ["a1", "a2"], ["s1", "s2"])
"""
export StrategyLossMatrix
export ObservationMatrix

export expected_strategy_loss
export bayes_strategy_loss

end
