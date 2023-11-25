

"""

"""
function expected_strategy_loss(SL::StrategyLossMatrix, O::ObservationMatrix)
    # expected loss of each strategy for each state
    D = zeros(length(SL.P), 2)
    for i in 1:length(SL.P)
        D[i,:] = sum(SL.SL[i]' .* O.O, dims=2)
    end
    return D
end


"""
L(s) = (1-w)L(θ₁, s) + wL(θ₂, s)   for prior (w, 1-w)
"""
function bayes_strategy_loss(SL::StrategyLossMatrix, i::Int, O::ObservationMatrix; prior=[1/3, 2/3])
    return sum(sum(SL.SL[i]' .* O.O, dims=2) .* prior)
end
