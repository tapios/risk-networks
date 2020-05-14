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
l = lambda x, k = 1.7, theta = 2: 2+np.random.gamma(k, theta)

# infectiousness duration outside hospitals
g = lambda x, k = 1.5, theta = 2: 1+np.random.gamma(k, theta)

# infectiousness duration in hospitals
gp = lambda x, k = 1.5, theta = 3: 1+np.random.gamma(k, theta)

# age structure King County, WA: https://datausa.io/profile/geo/king-county-wa#demographics
# we use the age classes: 0--19, 20--44, 45--64, 65--74, >= 75

kids         = 0.23
young_adults = 0.39
middle_aged  = 0.25
seniors      = 0.079
elderly      = 1 - seniors - middle_aged - young_adults - kids

age_classes = np.asarray([kids, young_adults, middle_aged, seniors, elderly])

# age distribution in working population: 20--44 and 45--64

age_classes_working = np.asarray([0.3876994201,0.2504385036])/sum([0.3876994201,0.2504385036])

# age-dependent hospitalization and recovery rates

fh = np.asarray([0.02, 0.17, 0.25, 0.35, 0.45])

fd = np.asarray([1e-15, 0.001, 0.005, 0.02, 0.05])

fdp = np.asarray([1e-15, 0.001, 0.01, 0.04, 0.1])

# hospitalization fraction (a...age class integers to refer to 0--19, 20--44, 45--64, 65--74, >= 75)

h = lambda a, beta = 4: np.random.beta(beta * fh[a] / (1-fh[a]), b = beta)

d = lambda a, beta = 4: np.random.beta(beta * fd[a] / (1-fd[a]), b = beta)

dp = lambda a, beta = 4: np.random.beta(beta * fdp[a] / (1-fdp[a]), b = beta)

# transmission

beta0  = 0.05
betap0 = 0.75 * beta0

beta_distr  = lambda x, beta = 4: beta0  * np.random.beta(beta * 0.05 / (1 - 0.05), b = beta)
betap_distr = lambda x, beta = 4: betap0 * np.random.beta(beta * 0.05 / (1 - 0.05), b = beta)

def temporalNetworkEpidemics(G, H, J, IC, return_statuses, deltat, T):
    """
    Simulation of SEIHRD dynamics on temporal networks.

    Parameters:
    ----------

    G (dictionary)          : dictionary with time stamps (seconds) and networks
    H (graph)               : spontaneous transitions
    J (graph)               : induced transitions
    IC (dictionary)         : initially infected nodes
    return_statuses (array) : specifying return compartments
    deltat (float)          : simulation time interval (in days)
    T (float)               : simulation period

    Returns:
    -------

    time_arr (array), states_arr (array): times and compartment values

    """

    # hospitalization dict (node x gets transferred to hospital node j)
    hospitalizations = {}

    time_arr = []
    states_arr = {}

    for s in return_statuses:
        states_arr['%s'%s] = []

    time_delta_arr = np.arange(0, 2+deltat, deltat)
    
    contact_reduction = 1
    
    for i in range(int(T/deltat)):
        
        day_time = time_delta_arr[i%len(time_delta_arr)]

        if day_time <= 0.5:
                        
            print(contact_reduction)
            
            J = inducedTransitions(G, beta0*np.sin(day_time*6)*contact_reduction, betap0*np.sin(day_time*6)*contact_reduction)
       
        else:
            
            J = inducedTransitions(G)

        res = EoN.Gillespie_simple_contagion(
                                G,                           # Contact network
                                H,                           # Spontaneous transitions (without any nbr influence)
                                J,                           # Neighbor induced transitions
                                IC,                          # Initial infected nodes
                                return_statuses,             
                                return_full_data = True,
                                tmax = deltat                
                            )

        times, states = res.summary()
        node_status = res.get_statuses(time = times[-1])

        contact_reduction = np.exp(-31*states['I'][-1]/len(G))

        for x in node_status.keys():

            # hopsitalization update

            # if node is outside hospital and has status "H"
            if x > init_placeholder and node_status[x] == 'H':
                
                hosp_flag = 0
                # check if hospital beds are available
                for j in range(init_placeholder):
                    
                    # hospitalization possible
                    if node_status[j] == 'P':
                        node_status[j] = 'H'
                        node_status[x] = 'P'
                        
                        # x gets transferred to j
                        hospitalizations[j] = x 
                        
                        IC[j] = 'H'
                        IC[x] = 'P'
                        
                        hosp_flag = 1
                        
                        break
                                    
            elif x <= init_placeholder and node_status[x] in ['D', 'R']:
                
                node_status[hospitalizations[x]] = node_status[x]
                node_status[x] = 'P'
                IC[x] = 'P'
                IC[hospitalizations[x]] = node_status[x]
                del hospitalizations[x]
                
            else:
                IC[x] = node_status[x]

            # need to add some random infections
            #if node_status[x] == 'S' and np.random.rand() < 0.1:
            #    IC[x] = 'I'
        #print(IC)
        
        time_arr.extend(times+i*deltat)
        for s in return_statuses:
            states_arr['%s'%s].extend(states['%s'%s])
        
        # stop simulation if no new infection occur
        if states['I'][-1] == 0 and states['E'][-1] == 0:
            break

    return np.asarray(time_arr), states_arr

def spontaneousTransitions(G):
    
    """
    spontaneous transitions of SEIHRD dynamics

    Parameters:
    G (graph): network

    Returns:
    H (graph): spontaneous transitions

    """
    
    gamma_arr = [1/g(1) for node in G.nodes()]    
    gammap_arr = [1/gp(1) for node in G.nodes()]    

    #h_arr = [h(np.random.choice(np.arange(5), p = age_classes)) for node in G.nodes()]
    #d_arr = [d(np.random.choice(np.arange(5), p = age_classes)) for node in G.nodes()]
    #dp_arr = [dp(np.random.choice(np.arange(5), p = age_classes)) for node in G.nodes()]
    
    h_arr = []
    d_arr = []
    dp_arr = []
    
    for i in range(len(G)):
        
        # health care workers have a different age structure: age_classes_working
        if i >= int(np.ceil(len(G)*0.005)) and i < int(np.ceil(len(G)*0.05)):
            h_arr.append(h(np.random.choice(np.arange(2)+1, p = age_classes_working)))
            d_arr.append(h(np.random.choice(np.arange(2)+1, p = age_classes_working)))
            dp_arr.append(h(np.random.choice(np.arange(2)+1, p = age_classes_working)))
         
        # for all other nodes, we use all age classes: age_classes
        else:
            h_arr.append(h(np.random.choice(np.arange(5), p = age_classes)))
            d_arr.append(d(np.random.choice(np.arange(5), p = age_classes)))
            dp_arr.append(dp(np.random.choice(np.arange(5), p = age_classes)))
    
    node_attribute_dict_E2I = {node: 1/l(node) for node in G.nodes()}
    node_attribute_dict_I2R = {node: (1-h_arr[node]-d_arr[node])*gamma_arr[node] for node in G.nodes()}
    node_attribute_dict_H2R = {node: (1-dp_arr[node])*gammap_arr[node] if node < int(np.ceil(len(G)*0.005)) else 0 for node in G.nodes()}
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
    H.add_node('P')
    H.add_edge('E', 'I', rate = 1, weight_label='expose2infect_weight')    # Latent period
    H.add_edge('I', 'R', rate = 1, weight_label='infect2recover_weight')   # Duration of infectiousness
    H.add_edge('H', 'R', rate = 1, weight_label='hospital2recover_weight') # Duration of infectiousness for hospitalized
    H.add_edge('I', 'H', rate = 1, weight_label='infect2hospital_weight')  # Hospitalization rate
    H.add_edge('I', 'D', rate = 1, weight_label='infect2death_weight')     # Death rate
    H.add_edge('H', 'D', rate = 1, weight_label='hospital2death_weight')   # Death rate for severe cases
    
    return H

def inducedTransitions(G, dbeta = 0, dbetap = 0):

    """
    induced transitions of SEIHRD dynamics

    Parameters:
    G (graph): network
    dbeta (float): delta beta
    dbetap (float): delta beta prime

    Returns:
    H (graph): spontaneous transitions

    """
    
    edge_attribute_dict_beta = {edge: beta_distr(1)+dbeta for edge in G.edges()}
    edge_attribute_dict_betap = {edge: betap_distr(1)+dbetap for edge in G.edges()}
    
    nx.set_edge_attributes(G, values=edge_attribute_dict_beta, name='beta_weight')
    nx.set_edge_attributes(G, values=edge_attribute_dict_betap, name='betap_weight')
    
    # Neighbor induced transitions.
    J = nx.DiGraph()
    J.add_edge(('I', 'S'), ('I', 'E'), rate = 1, weight_label='beta_weight')         # Transmission rate
    J.add_edge(('H', 'S'), ('H', 'E'), rate = 1, weight_label='betap_weight')        # Transmission rate
    
    return J
    
if __name__ == "__main__":
    
    arr = []
    # edge list data
    edge_list = np.loadtxt('../data/networks/edge_list_SBM_1e3.txt', dtype = int, comments = '#')

    # time step (duration) of each network snapshot in seconds

    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    
    IC = defaultdict(lambda: 'S')

    for i in range(int(0.5*len(G)),int(0.5*len(G))+10):
        IC[i] = 'I'
        
        
    # initial deaths (placeholder hospital)
    init_placeholder = int(np.ceil(len(G)*0.005))
    
    for i in range(init_placeholder):
        IC[i] = 'P'

    return_statuses = ('S', 'E', 'I', 'H', 'R', 'D', 'P')

    # transition networks

    H = spontaneousTransitions(G)
    J = inducedTransitions(G)  
    
    #edgeDict = G.edges(data=True)
    #print(edgeDict)    

    
    #for edge in G.edges():
    #    print(edge)
    #    print(G.get_edge_data(edge[0], edge[1]))
        
        
    # simulate dynamics
    times, states = temporalNetworkEpidemics(G, H, J, IC, return_statuses, deltat = 0.125, T = 100)

    tau = 5
    plt.figure()
    
    plt.plot(times, states['I'], '-o')
    
    plt.plot(times, 10*2.5**(times/tau))
    
    plt.yscale('log')
    
    plt.show()
    
    fig, axes = plt.subplots(1, 2, figsize = (15, 4))

    axes[0].plot(times, np.asarray(states['S'])/len(G), '-', label = 'Susceptible', color = 'C0')
    axes[0].plot(times, np.asarray(states['R'])/len(G), '-', label = 'Resistant', color = 'C4')
    axes[0].plot(times, (len(G)-np.asarray(states['S'])-init_placeholder)/len(G), '-', label = 'Total cases', color = 'C1')
    axes[0].legend(loc = 0)

    axes[1].plot(times, np.asarray(states['E'])/len(G), label = 'Exposed', color = 'C3')
    axes[1].plot(times, np.asarray(states['I'])/len(G), label = 'Infected', color = 'C1')
    axes[1].plot(times, np.asarray(states['H'])/len(G), label = 'Hospitalized', color = 'C2')
    axes[1].plot(times, (np.asarray(states['D'])-init_placeholder)/len(G), label = 'Death', color = 'C6')
    axes[1].legend(loc = 0)

    axes[0].set_xlabel('time [days]')
    axes[0].set_ylabel('fraction')
    axes[1].set_xlabel('time [days]')
    axes[1].set_ylabel('fraction')
    axes[0].set_ylim([0,0.2])
    axes[1].set_ylim([0,0.03])
    plt.tight_layout()
    plt.legend()
    plt.show()
