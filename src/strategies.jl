

"""

"""
function expected_strategy_loss(SL::StrategyLossMatrix, OM::ObservationMatrix)
    # expected loss of each strategy for each state
    D = zeros(length(SL.P), 2)
    for strategy in eachindex(SL.P)
        # indicator matrix size (|actions| x |observations|)
        AP = zeros(size(OM.states_observations)[2], size(L.actions)[1])

        for (i, val) in enumerate(SL.SL[strategy])
            AP[i, val] = 1
        end
    
        action_probabilities = OM.states_observations * AP        
        D[strategy,:] = sum(L.loss .* action_probabilities, dims=2)
    end
    return D
end


"""
L(s) = (1-w)L(θ₁, s) + wL(θ₂, s)   for prior (w, 1-w)
"""
function bayes_strategy_loss(SL::StrategyLossMatrix, i::Int, O::ObservationMatrix; prior=[1/3, 2/3])
    return sum(sum(SL.SL[i]' .* O.O, dims=2) .* prior)
end
