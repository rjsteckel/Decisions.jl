struct LossMatrix
    loss::NamedArray{Float64, 2}
    states::Vector{String}
    actions::Vector{String}

    function LossMatrix(losses::AbstractMatrix, states::Vector{String}, actions::Vector{String})
        @assert size(losses) == (length(states), length(actions)) "dimension check failed" 

        loss_named = NamedArray(
            Float64.(losses),
            names=(states, actions),
            dimnames=("states", "actions")
        )
        new(loss_named, states, actions)
    end
end

struct StrategyLossMatrix
    L::LossMatrix
    P::Vector{Tuple}
    A::Dict{String, Int}
    SL::Vector{Vector{Int}}

    function StrategyLossMatrix(L::LossMatrix, n_obs::Int)
        # all permutations of actions/observations (possible strategies)        
        it = IterTools.product(fill(L.actions, n_obs) ...)
        P = vec([reverse(p) for p in it])

        # action index
        A = Dict(zip(L.actions, 1:length(L.actions)))
        
        # Compute action index vectors for each strategy
        SL = [[A[s] for s in p] for p in P]
        new(L, P, A, SL)
    end
end

"""
    expected_strategy_loss(SL::StrategyLossMatrix, OM::ObservationMatrix)

Compute the risk (expected loss) for each strategy and each state using a tensor-based approach.
Returns a `NamedArray` where rows are named by the strategies (tuples in `SL.P`) and columns are named by states.
"""
function expected_strategy_loss(SL::StrategyLossMatrix, OM::ObservationMatrix)::NamedArray
    L = SL.L
    n_states = length(L.states)
    n_obs = size(OM.states_observations, 2)
    n_actions = length(L.actions)
    n_strategies = length(SL.P)

    # Tensor approach: V[θ, x, a] = P(x|θ) * L(θ, a)
    # Using broadcasting to avoid loops. .array access for speed.
    V = reshape(OM.states_observations.array, n_states, n_obs, 1) .* 
        reshape(L.loss.array, n_states, 1, n_actions)
    
    # Compute D[d, θ] = Σ_x V[θ, x, d(x)]
    D = zeros(n_strategies, n_states)
    for d in 1:n_strategies
        strategy_actions = SL.SL[d]
        for x in 1:n_obs
            a = strategy_actions[x]
            for θ in 1:n_states
                D[d, θ] += V[θ, x, a]
            end
        end
    end
    
    return NamedArray(D, (SL.P, L.states), ("strategies", "states"))
end

"""
    expected_strategy_loss(L::LossMatrix, OM::ObservationMatrix)

Compute the risk (expected loss) for each strategy and each state.
Returns a `NamedArray` where rows are named by the strategies and columns are named by states.
"""
function expected_strategy_loss(L::LossMatrix, OM::ObservationMatrix)::NamedArray
    n_obs = size(OM.states_observations, 2)
    SL = StrategyLossMatrix(L, n_obs)
    return expected_strategy_loss(SL, OM)
end

"""
    bayes_strategy_loss(SL::StrategyLossMatrix, i::Int, OM::ObservationMatrix; prior=nothing)

Compute the Bayes risk of strategy i under a given prior.
If prior is nothing, it assumes a uniform prior over states.
"""
function bayes_strategy_loss(SL::StrategyLossMatrix, i::Int, OM::ObservationMatrix; prior::Union{Nothing, Vector{<:Real}}=nothing)::Float64
    L = SL.L
    n_states = length(L.states)
    n_obs = size(OM.states_observations, 2)
    
    if isnothing(prior)
        prior = fill(1.0/n_states, n_states)
    end
    
    # Calculate risk R(θ, d_i) for all θ for strategy i
    strategy_actions = SL.SL[i]
    risk_i = zeros(n_states)
    for x in 1:n_obs
        a = strategy_actions[x]
        for θ in 1:n_states
            risk_i[θ] += OM.states_observations[θ, x] * L.loss[θ, a]
        end
    end
    
    return sum(risk_i .* prior)
end
