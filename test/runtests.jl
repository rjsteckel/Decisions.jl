using Decisions
using Test
using NamedArrays

@testset "Decisions.jl" begin
    # Original tests
    @testset "Original functionality" begin
        # losses for unknown states of nature and possible action combinations
        L = LossMatrix([0 1  3; 5 3 2], ["θ₁ (no rain)", "θ₂ (rain)"], ["action1", "action2", "action3"])

        # probabilities of observations given unknown state of nature
        probs = [0.60 0.25 0.15; 0.20 0.30 0.50]
        O = ObservationMatrix(probs, ["θ₁ (no rain)", "θ₂ (rain)"], ["x₁", "x₂", "x₃"])

        ES = expected_strategy_loss(L, O)
        @test ES[end,:] == [3.0; 2.0]

        # scatter(ES[:,1], ES[:,2])  # fig. 5.1
    end

    @testset "Fixed Bugs" begin
        # 1. Hardcoded dimension bug (3 states instead of 2)
        L3 = LossMatrix([0 1 3; 5 3 2; 1 1 1], ["θ₁", "θ₂", "θ₃"], ["a1", "a2", "a3"])
        probs3 = [0.6 0.2 0.2; 0.2 0.6 0.2; 0.2 0.2 0.6]
        O3 = ObservationMatrix(probs3, ["θ₁", "θ₂", "θ₃"], ["x1", "x2", "x3"])
        
        @testset "Fix: Hardcoded Dimension Bug" begin
            ES = expected_strategy_loss(L3, O3)
            @test size(ES, 2) == 3
            @test size(ES, 1) == 3^3 # 27 strategies
        end

        # 2. Strategy Enumeration Bug (Observations != Actions)
        # 2 actions, 3 observations -> should be 2^3 = 8 strategies
        L_small = LossMatrix([0 1; 1 0], ["θ₁", "θ₂"], ["a1", "a2"])
        O_large = ObservationMatrix([0.8 0.1 0.1; 0.1 0.1 0.8], ["θ₁", "θ₂"], ["x1", "x2", "x3"])
        
        @testset "Fix: Strategy Enumeration Bug" begin
            SL = StrategyLossMatrix(L_small, 3) 
            @test length(SL.P) == 8
        end

        # 3. Bayes Strategy Loss Bug
        @testset "Fix: Bayes Strategy Loss Bug" begin
            SL = StrategyLossMatrix(L_small, 3)
            bl = bayes_strategy_loss(SL, 1, O_large, prior=[0.5, 0.5])
            @test bl >= 0
            
            # Known value check for simple strategy
            # Strategy 1 is reverse(product(a1, a2)) -> likely always a1
            # If d(x) = a1, then risks are [0, 1] for θ₁, θ₂. 
            # Prior [0.5, 0.5] -> Bayes risk = 0.5
            @test bl ≈ 0.5
        end
    end
end
