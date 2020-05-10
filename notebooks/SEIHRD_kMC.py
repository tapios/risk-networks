import EoN
import numpy as np
import networkx as nx
from collections import defaultdict
from collections import OrderedDict
from scipy.stats import gamma
import matplotlib.pyplot as plt
import seaborn as sns
import random

# additional rate reference https://www.medrxiv.org/content/10.1101/2020.03.21.20040022v1.full.pdf

plt.style.use('seaborn-white')
sns.set_style("ticks")
sns.set_context("talk")

# latent period distribution
l = lambda x, k = 1.5, theta = 2: 2+np.random.gamma(k, theta)

# infectiousness duration outside hospitals
g = lambda x, k = 1.5, theta = 2: 1+np.random.gamma(k, theta)

# infectiousness duration in hospitals
gp = lambda x, k = 1.5, theta = 3: 1+np.random.gamma(k, theta)

# age structure Kings County, CA: https://datausa.io/profile/geo/kings-county-ca#demographics
# we use the age classes: 0--19, 20--44, 45--64, 65--74, >= 75

age_classes = np.asarray([0.312108117, 0.378902183, 0.214338507, 0.0548264451, 0.0398247471])

# age-dependent hospitalization and recovery rates

fh = np.asarray([0.01, 0.1, 0.2, 0.3, 0.3])

bh = np.asarray([0.02, 0.15, 0.15, 0.15, 0.45])

fd = np.asarray([0, 0.001, 0.005, 0.01, 0.03])

bd = np.asarray([0, 0.001, 0.005, 0.01, 0.02])

fdp = np.asarray([0, 0.001, 0.01, 0.02, 0.05])

bdp = np.asarray([0, 0.001, 0.01, 0.01, 0.15])

# hospitalization fraction (a...age class integers to refer to 0--19, 20--44, 45--64, 65--74, >= 75)
h = lambda a: fh[a]+np.random.uniform(low = 0, high = bh[a])

# mortality fraction outside hospitals
d = lambda a: fd[a]+np.random.uniform(low = 0, high = bd[a])

# mortality fraction in hospitals
dp = lambda a: fdp[a]+np.random.uniform(low = 0, high = bdp[a])

def temporal_network_epidemics(G, H, J, IC, return_statuses, deltat):

    """
    simulation of SEIHRD dynamics on temporal networks

    Parameters:
    G (dictionary): dictionary with time stamps (seconds) and networks
    H (graph): spontaneous transitions
    J (graph): induced transitions
    IC (dictionary): initially infected nodes
    return_statuses (array): specifying return compartments
    deltat (float): simulation time interval

    Returns:
    time_arr (array), states_arr (array): times and compartment values

    """

    res = EoN.Gillespie_simple_contagion(
                            G,                           # Contact network
                            H,                           # Spontaneous transitions (without any nbr influence)
                            J,                           # Neighbor induced transitions
                            IC,                          # Initial infected nodes
                            return_statuses,             
                            return_full_data = True,
                            tmax = 200                   # Contact network (division by 86400 because G time units are seconds)
                        )

    times, states = res.summary()
  
    return times, states

if __name__ == "__main__":

    # edge list data
    edge_list = np.loadtxt('../data/networks/edge_list_SBM_1e4.txt', dtype = int, comments = '#')

    # time step (duration) of each network snapshot in seconds

    G = nx.Graph()
    G.add_edges_from(edge_list)

    gamma_arr = [1/g(1) for node in G.nodes()]    
    gammap_arr = [1/gp(1) for node in G.nodes()]    

    h_arr = [h(np.random.choice(np.arange(5), p = age_classes)) for node in G.nodes()]
    d_arr = [d(np.random.choice(np.arange(5), p = age_classes)) for node in G.nodes()]
    dp_arr = [dp(np.random.choice(np.arange(5), p = age_classes)) for node in G.nodes()]
    
    node_attribute_dict_E2I = {node: 1/l(node) for node in G.nodes()}
    node_attribute_dict_I2R = {node: (1-h_arr[node]-d_arr[node])*gamma_arr[node] for node in G.nodes()}
    node_attribute_dict_H2R = {node: (1-dp_arr[node])*gammap_arr[node] for node in G.nodes()}
    node_attribute_dict_I2H = {node: h_arr[node]*gamma_arr[node] for node in G.nodes()}
    node_attribute_dict_I2D = {node: d_arr[node]*gamma_arr[node] for node in G.nodes()}
    node_attribute_dict_H2D = {node: dp_arr[node]*gammap_arr[node] for node in G.nodes()}
    
    nx.set_node_attributes(G, values=node_attribute_dict_E2I, name='expose2infect_weight')
    nx.set_node_attributes(G, values=node_attribute_dict_I2R, name='infect2recover_weight')
    nx.set_node_attributes(G, values=node_attribute_dict_H2R, name='hospital2recover_weight')
    nx.set_node_attributes(G, values=node_attribute_dict_I2H, name='infect2hospital_weight')
    nx.set_node_attributes(G, values=node_attribute_dict_I2D, name='infect2death_weight')
    nx.set_node_attributes(G, values=node_attribute_dict_H2D, name='hospital2death_weight')
    
    # Spontaneous transitions (without any nbr influence).
    H = nx.DiGraph()
    H.add_node('S')
    H.add_edge('E', 'I', rate = 1, weight_label='expose2infect_weight')    # Latent period
    H.add_edge('I', 'R', rate = 1, weight_label='infect2recover_weight')   # Duration of infectiousness
    H.add_edge('H', 'R', rate = 1, weight_label='hospital2recover_weight') # Duration of infectiousness for hospitalized
    H.add_edge('I', 'H', rate = 1, weight_label='infect2hospital_weight')  # Hospitalization rate
    H.add_edge('I', 'D', rate = 1, weight_label='infect2death_weight')     # Death rate
    H.add_edge('H', 'D', rate = 1, weight_label='hospital2death_weight')   # Death rate for severe cases


    # transmission rates
    beta = 0.02
    betap = 0.75*beta
    
    # Neighbor induced transitions.
    J = nx.DiGraph()
    J.add_edge(('I', 'S'), ('I', 'E'), rate = beta)         # Transmission rate
    J.add_edge(('H', 'S'), ('H', 'E'), rate = betap)        # Transmission rate

    IC = defaultdict(lambda: 'S')

    for i in range(10):
        IC[i] = 'I'

    return_statuses = ('S', 'E', 'I', 'H', 'R', 'D')

    # simulate dynamics
    times, states = temporal_network_epidemics(G, H, J, IC, return_statuses, deltat = 100)

    
    tau = 5
    plt.figure()
    
    plt.plot(times, states['I'])
    
    plt.plot(times, 10*2.5**(times/tau))
    
    plt.yscale('log')
    
    plt.show()
    
    fig, axes = plt.subplots(1, 2, figsize = (15, 4))

    axes[0].plot(times, states['S'], label = 'Susceptible', color = 'C0')
    axes[0].plot(times, states['R'], label = 'Resistant', color = 'C4')
    axes[0].plot(times, states['I'], label = 'Infected', color = 'C1')
    axes[0].legend(loc = 0)

    axes[1].plot(times, states['E'], label = 'Exposed', color = 'C3')
    axes[1].plot(times, states['I'], label = 'Infected', color = 'C1')
    axes[1].plot(times, states['H'], label = 'Hospitalized', color = 'C2')
    axes[1].plot(times, states['D'], label = 'Death', color = 'C6')
    axes[1].legend(loc = 0)

    axes[0].set_xlabel('time [days]')
    axes[0].set_ylabel('number')
    axes[1].set_xlabel('time [days]')
    axes[1].set_ylabel('number')
    plt.tight_layout()
    plt.legend()
    plt.show()
