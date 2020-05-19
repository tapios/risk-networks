#import EoN
from simulation import Gillespie_simple_contagion
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import random

# additional rate reference https://www.medrxiv.org/content/10.1101/2020.03.21.20040022v1.full.pdf

plt.style.use('seaborn-white')
sns.set_style("ticks")
sns.set_context("talk")

if __name__ == "__main__":
    
    N = int(1e3)
    G = nx.fast_gnp_random_graph(N, 5./(N-1))
    
    node_attribute_dict = {node: 0.5+random.random() for node in G.nodes()}
    edge_attribute_dict = {edge: 0.5+random.random() for edge in G.edges()}
    
    nx.set_node_attributes(G, values=node_attribute_dict, name='expose2infect_weight')
    nx.set_edge_attributes(G, values=edge_attribute_dict, name='transmission_weight')
    
    # Parameters
    
    # S-->E rate
    beta = 0.06  
    
    # E-->I rate
    sigma =  1/3.5      
    
    # I-->R rate
    gamma = 1/13.7
    
    # I-->D rate and I-->H rate
    mu, delta = np.linalg.solve(np.array([[1 - 0.01,  -0.01],[-0.15, 1 - 0.15]]), 
                                np.array([[0.01 *(gamma)],[0.15 *(gamma)]]))
    # H-->R rate
    gammap = (1/(1/gamma + 7.0))
    
    print(gammap)
    # H-->D rate
    mup = (0.1/(1-.1))*gammap
    
    print(mup)
    # Spontaneous transitions (without any nbr influence). 
    H = nx.DiGraph()
    H.add_node('S')
    H.add_edge('E', 'I', rate = sigma)                              # Latent period
    #H.add_edge('A', 'I', rate = alpha)                             # Asymptomatic period
    H.add_edge('I', 'R', rate = gamma)                              # Duration of infectiousness
    H.add_edge('H', 'R', rate = gammap)                             # Duration of infectiousness for hospitalized
    H.add_edge('I', 'H', rate = delta[0])                           # Hospitalization rate
    H.add_edge('I', 'D', rate = mu[0])                              # Death rate 
    H.add_edge('H', 'D', rate = gammap)                             # Death rate for severe cases
                           
    
    # Neighbor induced transitions.
    J = nx.DiGraph()
    J.add_edge(('I', 'S'), ('I', 'E'), rate = beta)         # Transmission rate
    J.add_edge(('H', 'S'), ('H', 'E'), rate = beta)         # Transmission rate
    #J.add_edge(('A', 'S'), ('A', 'E'), rate = beta)         # Transmission rate
    #J.add_edge(('E', 'S'), ('E', 'E'), rate = .2 * beta)   # Transmission rate
    
    IC = defaultdict(lambda: 'S')
    for node in range(200):
        IC[node] = 'I'
    
    return_statuses = ('S', 'E', 'I', 'H', 'R', 'D')
    
    res = Gillespie_simple_contagion(
                        G,                           # Contact network.
                        H,                           # Spontaneous transitions (without any nbr influence).
                        J,                           # Neighbor induced transitions.
                        IC,                          # Initial infected nodes
                        return_statuses,             # 
                        return_full_data = True,
                        tmax = float('Inf')          # Contact network
                    )
    
    times, states = res.summary()
    
    fig, axes = plt.subplots(1, 2, figsize = (15, 4))
    
    axes[0].plot(times, states['S'], label = 'Susceptible', color = 'C0')
    axes[0].plot(times, states['R'], label = 'Resistant', color = 'C4')
    axes[0].plot(times, states['I'], label = 'Infected', color = 'C1')
    axes[0].legend()
    
    axes[1].plot(times, states['E'], label = 'Exposed', color = 'C3')
    axes[1].plot(times, states['I'], label = 'Infected', color = 'C1')
    axes[1].plot(times, states['H'], label = 'Hospitalized', color = 'C2')
    axes[1].plot(times, states['D'], label = 'Death', color = 'C6')
    axes[1].legend()
    
    plt.legend()
    plt.show()