using Decisions
using Test

@testset "Decisions.jl" begin
    # losses for unknown states of nature and possible action combinations
    L = LossMatrix([0 1  3; 5 3 2], ["θ₁ (no rain)", "θ₂ (rain)"], ["action1", "action2", "action3"])

    # probabilities of observations given unknown state of nature
    probs = [0.60 0.25 0.15; 0.20 0.30 0.50]
    O = ObservationMatrix(probs, ["θ₁ (no rain)", "θ₂ (rain)"], ["x₁", "x₂", "x₃"])

    ES = expected_strategy_loss(L, O)
    @test ES[end,:] == [3.0; 2.0]

    # using Plots
    # scatter(ES[:,1], ES[:,2]; xlims=[0,4], ylims=[0,6])
end








