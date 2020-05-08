using DifferentialEquations, PyPlot, InfectionNetworks, LightGraphs, SimpleWeightedGraphs

using Distributions

N = 200

initial_state = zeros(2, N)
initial_state[2, :] .= 0.01 * rand(N)            # Sprinkle some infected people around
initial_state[1, :] .= N .- initial_state[2, :] # Nobody is recovered.

# Local banded network with random nonlocal connections
network = SimpleWeightedGraph(N)

function connect_randomly!(network, p)
    for i = 1:N
        for j = i+1:N
            if rand() < p
                add_edge!(network, i, j, rand())
            end
        end
    end

    return nothing
end

function add_cluster!(network, i, clustersize)
    for j = i-clustersize : i + clustersize
        i != j && add_edge!(network, i, j, exp(- (i-j)^2 / (2 * (clustersize/2)^2)))
    end

    return nothing
end

connect_randomly!(network, 0.01)

# Clusters
Nc = 4
for i = 1:20
    add_cluster!(network, rand(DiscreteUniform(Nc, N-Nc)), Nc)
end

problem = NetworkSIRProblem(initial_state, (0.0, 100000.0),
                                      network = network,
                            transmission_rate = 1e-9,
                                recovery_rate = 1e-6)
solution = solve(problem)

t, S, I, R = extract(solution)

close("all")
fig, axs = subplots(ncols=2, figsize=(20, 12))

sca(axs[1])
pcolormesh(network.weights, vmin=0, vmax=1, cmap="binary")
title("Network connectivity")
xlabel(L"i")
ylabel(L"j")

sca(axs[2])
pcolormesh(I)
title("Infection as function of time")

[ax.set_aspect(1) for ax in axs]
