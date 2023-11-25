module Decisions

using NamedArrays
using IterTools

include("losses.jl")
include("observations.jl")
include("strategies.jl")


export LossMatrix
export StrategyLossMatrix
export ObservationMatrix

export expected_strategy_loss
export bayes_strategy_loss

end
