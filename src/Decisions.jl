module Decisions

using NamedArrays

include("losses.jl")
include("observations.jl")

export LossMatrix
export StrategyLossMatrix
export ObservationMatrix

end
