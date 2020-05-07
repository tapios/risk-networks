using DifferentialEquations, PyPlot, InfectionNetworks, LightGraphs, SimpleWeightedGraphs

using Distributions

function random_network_and_state(N; p=0.1)

    initial_state = zeros(2, N)
    initial_state[2, :] .= 0.01 * rand(N)            # Sprinkle some infected people around
    initial_state[1, :] .= N .- initial_state[2, :] # Nobody is recovered.
    
    # Local banded network with random nonlocal connections
    network = SimpleWeightedGraph(N)
    
    for i = 1:N
        for j = i+1:N
            if rand() < p
                add_edge!(network, i, j, rand())
            end
        end
    end

    #for i = 1:N
    #    for j = i+1:N
    #        add_edge!(network, i, j, 1)
    #    end
    #end

    return network, initial_state
end

function solve_network_problem(initial_state, network, T, β, γ)

    problem = NetworkSIRProblem(initial_state, (0.0, T),
                                                  network = network,
                                        transmission_rate = β,
                                            recovery_rate = γ)
    
    solution = solve(problem, reltol=1e-9)

    t, S, I, R = extract(solution)

    return t, S, I, R
end

function solve_network_forward_backward(initial_state, network, T, β, γ)

    forward_problem = NetworkSIRProblem(initial_state, (0.0, T),
                                                  network = network,
                                        transmission_rate = β,
                                            recovery_rate = γ)
    
    forward_solution = solve(forward_problem, reltol=1e-9, 
                             saveat=range(0.1T, stop=T, length=1000))

    backward_problem = NetworkSIRProblem(forward_solution.u[end], (0.0, 0.9T),
                                                   network = network,
                                         transmission_rate = -β,
                                             recovery_rate = -γ)
     
    backward_solution = solve(backward_problem, reltol=1e-9,
                              saveat=range(0.0, stop=0.9T, length=1000))

    names = (:t, :S, :I, :R)

    return (
            NamedTuple{names}(extract(forward_solution)), 
            NamedTuple{names}(extract(backward_solution))
           )
end


N = 1000
T = 1e10
β = 1e-5 / N^2 
γ = 1e-9

network, initial_state = random_network_and_state(N; p=0.001)

fwd, bwd = solve_network_forward_backward(initial_state, network, T, β, γ)

error = (abs.(fwd.I[1, :] .- reverse(bwd.I[1, :]))) ./ abs.(fwd.I[1, :])

close("all")
plot(fwd.t, error)
#plot(fwd.t,               fwd.I[1, :],       label="forward")
#plot((fwd.t[end].-bwd.t), bwd.I[1, :], "--", label="backward")
legend()

fig, axs = subplots(ncols=3, figsize=(20, 12))

sca(axs[1])
pcolormesh(network.weights, vmin=0, vmax=1, cmap="binary")
title("Network connectivity")
xlabel(L"i")
ylabel(L"j")

sca(axs[2])
pcolormesh(fwd.I)
title("Infection as function of time")

sca(axs[3])
pcolormesh(bwd.I)
title("Infection as function of time")

[ax.set_aspect(1) for ax in axs]
