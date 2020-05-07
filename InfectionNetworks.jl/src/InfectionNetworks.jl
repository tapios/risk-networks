module InfectionNetworks

export
    Network,
    Infection,
    NetworkSIRParameters,
    network_sir!,
    NetworkSIRProblem,
    connect_locally!,
    connect_randomly!,
    extract

using DifferentialEquations: ODEProblem

struct Infection{T, A}
    transmission_rate :: T
             pressure :: A # Infectious pressure on node i
             recovery :: A # Recovery rate of node i
end

Infection(N; transmission_rate=0.1, recovery_rate=0.1) =
    Infection(transmission_rate, zeros(N), recovery_rate .* ones(N))

"""
    network_SIR!(∂t_states, states, parameters, time)

Implements a Susceptible, Infected, Resistant model on a network of the form

    ``∂t Sᵢ = -βᵢ Sᵢ``
    ``∂t Iᵢ = βᵢ Sᵢ - γᵢ I``

where ``βᵢ`` is the infection pressure on node i, and ``γᵢ`` is the recovery
rate of node i.

The infection pressure is modeled as

    ``βᵢ = wᵢⱼ βᵢⱼ Iⱼ``

Notes:

    * parameters.network contains the network parameters:
        - parameters.network.connectivity: connectivity of the network (weights w₁ⱼ between 0 and 1)
        - parameters.network.transmission_rate: transmission rates of the network from i to j
    
    * parameters.infection.pressure carries scratch space for the infection pressure 
        (computed dynamically from the network parameters)
    
    * parameters.infection.recovery stores recovery rates for node i
"""
function network_SIR!(∂t_states, states, parameters, time)
    S = view(states, 1, :)
    I = view(states, 2, :)

    parameters.infection.pressure .= parameters.network.weights * I
    parameters.infection.pressure .*= parameters.infection.transmission_rate

    @views @. ∂t_states[1, :] = - parameters.infection.pressure * S
    @views @. ∂t_states[2, :] =   parameters.infection.pressure * S - parameters.infection.recovery * I
end


function NetworkSIRParameters(network; transmission_rate=0.1, recovery_rate=0.1)
    population = size(network)[1]
    return (      network = network,
                infection = Infection(population, transmission_rate=transmission_rate, 
                                      recovery_rate=recovery_rate),
               population = population
           )
end

function NetworkSIRProblem(initial_state, time_span; network, parameters_kwargs...)
    parameters = NetworkSIRParameters(network; parameters_kwargs...)
    return ODEProblem(network_SIR!, initial_state, time_span, parameters)
end

function extract(solution)
    nt = length(solution.t)
    N = solution.prob.p.population
    S = zeros(N, nt)
    I = zeros(N, nt)

    for j = 1:nt
        for i = 1:N
            S[i, j] = solution.u[j][1, i]
            I[i, j] = solution.u[j][2, i]
        end
    end

    R = @. N - S - I

    return solution.t, S, I, R
end

end # module
