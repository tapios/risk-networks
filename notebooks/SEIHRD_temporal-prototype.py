import EoN
import numpy as np
import networkx as nx
from collections import defaultdict
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import random

# additional rate reference https://www.medrxiv.org/content/10.1101/2020.03.21.20040022v1.full.pdf

plt.style.use('seaborn-white')
sns.set_style("ticks")
sns.set_context("talk")

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

    time_arr = []
    states_arr = {}
    for s in return_statuses:
        states_arr['%s'%s] = []

    for i in G.keys():

        # transition rates are given in units 1/day,
        # so we simulate in days as time units

        # float('Inf')

        #plt.close('all')
        #nx.draw(G[i], pos=nx.spring_layout(G[i]), node_size=3)
        #plt.show()

        # add all previously "active" nodes to current network

        print(i, len(G[i]))

        res = EoN.Gillespie_simple_contagion(
                                G[i],                        # Contact network
                                H,                           # Spontaneous transitions (without any nbr influence)
                                J,                           # Neighbor induced transitions
                                IC,                          # Initial infected nodes
                                return_statuses,             #
                                return_full_data = True,
                                tmax = deltat/86400          # Contact network (division by 86400 because G time units are seconds)
                            )

        times, states = res.summary()
        node_status = res.get_statuses(time = times[-1])

        for x in node_status.keys():

            IC[x] = node_status[x]

            # need to add some random infections
            #if node_status[x] == 'S' and np.random.rand() < 0.1:
            #    IC[x] = 'I'
        #print(IC)

        time_arr.extend(times+i/86400)
        for s in return_statuses:
            states_arr['%s'%s].extend(states['%s'%s])



    return time_arr, states_arr


def temporal_network(edge_list, deltat):
    """
    temporal network construction

    Parameters:
    edge_list (array): edge list (1st column: time stamp UNIX format, 2nd-3rd columnm: edge i <-> j)
    deltat (float): time step (duration) of each network snapshot in seconds

    Returns:
    Gord: dictionary with time stamps (seconds) and networks

    """
    
    G = {}

    G1 = nx.Graph()

    T0 = edge_list[0][0]

    T = edge_list[0][0]

    nodes = edge_list[:,1]
    nodes = np.append(nodes, edge_list[:,2])
    nodes = set(nodes)

    Gnodes = nx.Graph()
    Gnodes.add_nodes_from(nodes)

    for i in range(len(edge_list)):

        if edge_list[i][0] <= T + deltat:
            G1.add_nodes_from(nodes)
            G1.add_edge(edge_list[i][1],edge_list[i][2])
        else:

            if len(G1):
                G[(T-T0)] = G1
                G1 = nx.Graph()
            else:
                G[(T-T0)] = Gnodes
            T += deltat

    Gord = OrderedDict(sorted(G.items()))
    return Gord

if __name__ == "__main__":

    #N = int(1e4)
    #G = nx.fast_gnp_random_graph(N, 5./(N-1))

    #node_attribute_dict = {node: 0.5+random.random() for node in G.nodes()}
    #edge_attribute_dict = {edge: 0.5+random.random() for edge in G.edges()}

    #nx.set_node_attributes(G, values=node_attribute_dict, name='expose2infect_weight')
    #nx.set_edge_attributes(G, values=edge_attribute_dict, name='transmission_weight')


    # edge list data
    edge_list = np.loadtxt('../data/networks/thiers_2012.csv', usecols = [0,1,2], dtype = int)

    # time step (duration) of each network snapshot in seconds
    deltat = 3600

    G = temporal_network(edge_list, deltat)

    density = np.asarray([2*len(x.edges())/(len(x.nodes())*(len(x.nodes())-1)) for x in G.values()])
    time = np.asarray([x for x in G.keys()])/(3600)
    
    lamb = lambda t: np.max(np.asarray([0.1,1-np.cos(np.pi*t/24)**4]))
    print(time)
    plt.figure()
    plt.plot(time+6, density/max(density))
    plt.plot(time[time <= 24],[lamb(t) for t in time[time <= 24]])
    plt.xlim([0,25])
    plt.ylim([0,1])
    plt.xlabel(r'time [hours]')
    plt.ylabel(r'normalized edge density')
    plt.tight_layout()
    plt.show()

    # Parameters

    # S-->E rate
    beta = 0.5

    # E-->I rate
    sigma =  1/3.5

    # I-->R rate
    gamma = 1/13.7

    # I-->D rate and I-->H rate
    mu, delta = np.linalg.solve(np.array([[1 - 0.01,  -0.01],[-0.15, 1 - 0.15]]),
                                np.array([[0.01 *(gamma)],[0.15 *(gamma)]]))
    # H-->R rate
    gammap = (1/(1/gamma + 7.0))

    # H-->D rate
    mup = (0.1/(1-.1))*gammap

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
    #J.add_edge(('A', 'S'), ('A', 'E'), rate = beta)        # Transmission rate
    #J.add_edge(('E', 'S'), ('E', 'E'), rate = .2 * beta)   # Transmission rate

    IC = defaultdict(lambda: 'S')

    initial_nodes = list(G[0].nodes())
    for i in range(30):
        IC[initial_nodes[i]] = 'I'

    return_statuses = ('S', 'E', 'I', 'H', 'R', 'D')

    # simulate dynamics
    times, states = temporal_network_epidemics(G, H, J, IC, return_statuses, deltat)

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
