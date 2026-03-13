"""
    LossMatrix(losses, states, actions)

A structure representing a loss matrix L(θ, a) where θ are states and a are actions.
Uses `NamedArrays` for labeled access.
"""
struct LossMatrix
    loss::NamedArray{Float64, 2}
    states::Vector{String}
    actions::Vector{String}

    function LossMatrix(losses::AbstractMatrix{<:Real}, states::AbstractVector{<:AbstractString}, actions::AbstractVector{<:AbstractString})
        @assert size(losses) == (length(states), length(actions)) "Dimension mismatch: size(losses) must be (length(states), length(actions))" 

        loss_named = NamedArray(
            Float64.(losses),
            names=(String.(states), String.(actions)),
            dimnames=("states", "actions")
        )
        new(loss_named, String.(states), String.(actions))
    end
end

"""
    StrategyLossMatrix(L, n_obs)

A structure to store all possible strategies and their precomputed action indices.
A strategy is a mapping from observations to actions.
If there are |A| actions and |X| observations, there are |A|^|X| possible strategies.
"""
struct StrategyLossMatrix
    L::LossMatrix
    P::Vector{<:Tuple}
    A::Dict{String, Int}
    SL::Matrix{Int} # Matrix of size (n_strategies, n_obs)

    function StrategyLossMatrix(L::LossMatrix, n_obs::Int)
        # all permutations of actions/observations (possible strategies)        
        it = Iterators.product(fill(L.actions, n_obs) ...)
        P = vec([reverse(p) for p in it])

        # action index mapping
        A = Dict(zip(L.actions, eachindex(L.actions)))
        
        # Compute action index matrix for each strategy
        n_strategies = length(P)
        SL = Matrix{Int}(undef, n_strategies, n_obs)
        for d in 1:n_strategies
            for x in 1:n_obs
                SL[d, x] = A[P[d][x]]
            end
        end
        new(L, P, A, SL)
    end
end

"""
    expected_strategy_loss(SL::StrategyLossMatrix, OM::ObservationMatrix)

Compute the risk (expected loss) for each strategy and each state using a tensor-based approach.
Returns a `NamedArray` where rows are named by the strategies (tuples) and columns are named by states.
"""
function expected_strategy_loss(SL::StrategyLossMatrix, OM::ObservationMatrix)::NamedArray
    L = SL.L
    n_states = length(L.states)
    n_obs = size(OM.states_observations, 2)
    n_actions = length(L.actions)
    n_strategies = length(SL.P)

    # Tensor approach: V[θ, x, a] = P(x|θ) * L(θ, a)
    # Using broadcasting for speed.
    V = reshape(OM.states_observations.array, n_states, n_obs, 1) .* 
        reshape(L.loss.array, n_states, 1, n_actions)
    
    # Compute D[d, θ] = Σ_x V[θ, x, d(x)]
    # D is (n_strategies, n_states). In Julia, first index varies fastest.
    D = zeros(n_strategies, n_states)
    
    # Optimized loop order for memory locality (column-major)
    for θ in 1:n_states
        for x in 1:n_obs
            # Pre-index V for this state and observation
            Vx = @view V[θ, x, :]
            for d in 1:n_strategies
                a = SL.SL[d, x]
                D[d, θ] += Vx[a]
            end
        end
    end
    
    return NamedArray(D, (SL.P, L.states), ("strategies", "states"))
end

"""
    expected_strategy_loss(L::LossMatrix, OM::ObservationMatrix)

Compute the risk (expected loss) for each strategy and each state.
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
function bayes_strategy_loss(SL::StrategyLossMatrix, i::Int, OM::ObservationMatrix; prior::Union{Nothing, AbstractVector{<:Real}}=nothing)::Float64
    L = SL.L
    n_states = length(L.states)
    n_obs = size(OM.states_observations, 2)
    
    p = isnothing(prior) ? fill(1.0/n_states, n_states) : Float64.(prior)
    
    # Calculate risk R(θ, d_i) for all θ for strategy i
    risk_i = zeros(n_states)
    for x in 1:n_obs
        a = SL.SL[i, x]
        for θ in 1:n_states
            risk_i[θ] += OM.states_observations[θ, x] * L.loss[θ, a]
        end
    end
    
    return sum(risk_i .* p)
end
