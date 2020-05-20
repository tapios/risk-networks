#import EoN
from simulation import Gillespie_simple_contagion
import numpy as np
import networkx as nx
from collections import defaultdict
from collections import OrderedDict
from scipy.stats import gamma
import matplotlib.pyplot as plt
import seaborn as sns
import random

np.random.seed(1123)
random.seed(1123)
    
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

age_classes = np.asarray([0.2298112587,0.3876994201,0.2504385036,0.079450985,0.0525998326])

# age distribution in working population: 20--44 and 45--64

age_classes_working = np.asarray([0.3876994201,0.2504385036])/sum([0.3876994201,0.2504385036])

# age-dependent hospitalization and recovery rates

fh = np.asarray([0.02, 0.17, 0.25, 0.35, 0.45])

fd = np.asarray([0.001, 0.001, 0.005, 0.02, 0.05])

fdp = np.asarray([0.001, 0.001, 0.01, 0.04, 0.1])

# hospitalization fraction (a...age class integers to refer to 0--19, 20--44, 45--64, 65--74, >= 75)

h = lambda a, beta = 4: np.random.beta(beta*fh[a]/(1-fh[a]), b = beta)

d = lambda a, beta = 4: np.random.beta(beta*fd[a]/(1-fd[a]), b = beta)

dp = lambda a, beta = 4: np.random.beta(beta*fdp[a]/(1-fdp[a]), b = beta)

# transmission

beta0 = 0.14

alpha_hosp = 0.25

def temporalNetworkEpidemics(G, H, IC, return_statuses, deltat, T):

    """
    simulation of SEIHRD dynamics on temporal networks

    Parameters:
    G (dictionary): dictionary with time stamps (seconds) and networks
    H (graph): spontaneous transitions
    IC (dictionary): initially infected nodes
    return_statuses (array): specifying return compartments
    deltat (float): simulation time interval (in days)
    T (float): simulation period

    Returns:
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
    
    edge_dict = edgeRenewal(edge_list)
    beta_dict, betap_dict = betaAverage(G, edge_dict, deltat_kMC)
    
    quarantine_flag = 0
    
    for i in range(int(T/deltat)):           
    
        J = inducedTransitions(beta_dict[i%len(beta_dict.keys())], betap_dict[i%len(beta_dict.keys())])#beta_dict,betap_dict)

        res = Gillespie_simple_contagion(
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
        
        print(times[-1]+i*deltat, states['S'][-1], states['E'][-1], states['I'][-1], states['H'][-1], states['R'][-1], states['D'][-1], states['S'][-1]+states['E'][-1]+states['I'][-1]+states['H'][-1]+states['R'][-1]+states['D'][-1])
        
        # contact reduction
        if states['H'][-1]/len(G) >= 0.003 and quarantine_flag == 0:  
            quarantine_flag = 1
            edge_dict = edgeRenewal(edge_list, lamb_min = 5/24, lamb_max = 0.2*22/24)
            beta_dict, betap_dict = betaAverage(G, edge_dict, deltat_kMC)
              
        time_arr.extend(times+i*deltat)
        for s in return_statuses:
            states_arr['%s'%s].extend(states['%s'%s])
            
        for x in node_status.keys():

            # hopsitalization update

            # if node is outside hospital and has status "H"
            if x >= init_placeholder and node_status[x] == 'H':
                
                hosp_flag = 0
                # check if hospital beds are available
                for j in range(init_placeholder):
                    
                    # hospitalization possible
                    if node_status[j] == 'P':
                        node_status[j] = 'H'
                        node_status[x] = 'P'
                        
                        # x gets transferred to j
                        hospitalizations[j] = np.copy(x) 
                        
                        IC[j] = node_status[j]
                        IC[x] = node_status[x]
                        
                        # change relevant node attributes of G
                        node_attribute_dict_H2R[j] = np.copy(node_attribute_dict_H2R[x])
                        node_attribute_dict_H2D[j] = np.copy(node_attribute_dict_H2D[x])
                        
                        nx.set_node_attributes(G, values=node_attribute_dict_H2R, name='hospital2recover_weight')
                        nx.set_node_attributes(G, values=node_attribute_dict_H2D, name='hospital2death_weight')
                                                
                        hosp_flag = 1
                        
                        break
                 
                # if node cannot be hospitalized
                if hosp_flag == 0:
                    IC[x] = node_status[x]
            
            # if node is in hospital and resolved                        
            elif x < init_placeholder and node_status[x] in ['D', 'R']:
                                
                node_status[int(hospitalizations[x])] = node_status[x]
                IC[int(hospitalizations[x])] = node_status[x]
                node_status[x] = 'P'
                IC[x] = node_status[x]
                del hospitalizations[x]
                
            elif x >= init_placeholder and node_status[x] != 'H':
                IC[x] = node_status[x]
            
            elif x < init_placeholder and node_status[x] not in ['D', 'R']:
                IC[x] = node_status[x]

            # need to add some random infections
            #if node_status[x] == 'S' and np.random.rand() < 0.1:
            #    IC[x] = 'I'
        #print(IC)
        
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
            age = np.random.choice(np.arange(2)+1
            h_arr.append(h(age, p = age_classes_working)))
            d_arr.append(h(age, p = age_classes_working)))
            dp_arr.append(h(age, p = age_classes_working)))
         
        # for all other nodes, we use all age classes: age_classes
        else:
            age = np.random.choice(np.arange(2)+1
            h_arr.append(h(age, p = age_classes)))
            d_arr.append(d(age, p = age_classes)))
            dp_arr.append(dp(age, p = age_classes)))
    
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
    H.add_node('P')
    H.add_edge('E', 'I', rate = 1, weight_label='expose2infect_weight')    # Latent period
    H.add_edge('I', 'R', rate = 1, weight_label='infect2recover_weight')   # Duration of infectiousness
    H.add_edge('H', 'R', rate = 1, weight_label='hospital2recover_weight') # Duration of infectiousness for hospitalized
    H.add_edge('I', 'H', rate = 1, weight_label='infect2hospital_weight')  # Hospitalization rate
    H.add_edge('I', 'D', rate = 1, weight_label='infect2death_weight')     # Death rate
    H.add_edge('H', 'D', rate = 1, weight_label='hospital2death_weight')   # Death rate for severe cases
    
    return H, node_attribute_dict_E2I, node_attribute_dict_I2R, node_attribute_dict_H2R, node_attribute_dict_I2H, node_attribute_dict_I2D, node_attribute_dict_H2D

def inducedTransitions(beta_dict, betap_dict):

    """
    induced transitions of SEIHRD dynamics

    Parameters:
    beta_dict (dictionary): edges and corresponding beta values
    betap_dict (dictionary): edges and corresponding beta values

    Returns:
    J (graph): inducedTransitions

    """

    nx.set_edge_attributes(G, values=beta_dict, name='beta_weight')
    nx.set_edge_attributes(G, values=betap_dict, name='betap_weight')
    
    # Neighbor induced transitions.
    J = nx.DiGraph()
    J.add_edge(('I', 'S'), ('I', 'E'), rate = 1, weight_label='beta_weight')         # Transmission rate
    J.add_edge(('H', 'S'), ('H', 'E'), rate = 1, weight_label='betap_weight')        # Transmission rate
    
    return J


def edgeRenewal(edge_list, lamb_min = 5/24, lamb_max = 22/24):
    
    """
    temporal edge activation/deactivation
    
    Parameters:
    edge_list: list
    lamb_max: 

    Returns:
    edge_dict (dictionary): index, time, and corresponding edge list

    """
    
    # duration rate (unit: 1/h)
    muc = 6    
    
    # mean contact rate (unit t: h)
    lamb = lambda t: np.max([lamb_min, lamb_max*(1-np.cos(np.pi*t/24)**2)])
    
    
    # initial active and inactive lists
    initial_active_frac = 0.034
    
    active_list = [tuple(edge) for edge in edge_list if np.random.random() < initial_active_frac]
    
    inactive_list = [tuple(edge) for edge in edge_list if tuple(edge) not in active_list]
    
    edge_dict = {}

    edge_dict[0] = {'time': 0, 'edge_list': [x for x in active_list]}
    
    # loop over activation/deactivation processes
    t = 0
    
    it = 0
        
    dt = 0.1
    
    tmeas = dt
    
    print("w_ji generation started")

    while t <= 24:
        
        # rate process 1: deactivation
        Q1 = muc*len(active_list)
        
        # rate process 3: activation
        Q2 = lamb(t)*len(inactive_list)
        
        # deactivation
        if np.random.random() < Q1/(Q1+Q2):
            ind = np.random.randint(len(active_list))
            inactive_list.append(active_list[ind])
            del active_list[ind]
            
        # activation
        else:
            ind = np.random.randint(len(inactive_list))
            active_list.append(inactive_list[ind])
            del inactive_list[ind]
    
        t += -np.log(np.random.random())/(Q1+Q2)
                
        
        if t >= tmeas:
            it += 1
            edge_dict[it] = {'time': t, 'edge_list': [x for x in active_list]}
            tmeas += dt

    print("w_ji generation ended")

#    plt.figure()
#    plt.plot([edge_dict[x]['time'] for x in range(len(edge_dict))], [len(edge_dict[x]['edge_list'])/len(edge_list) for x in range(len(edge_dict))])
#    plt.xlabel('hours')
#    plt.ylabel('fraction of active edges')
#    plt.xlim([0,25])
#    plt.ylim([0,0.15])
#    plt.tight_layout()
#    plt.show()

    return edge_dict
    
def betaAverage(G, edge_dict, deltat_kMC):
    
    """
    temporal average of beta    
    
    Parameters:
    G (graph): graph object
    edge_dict (dictionary): index, time, and corresponding active edge list
    deltat_kMC: kMC time step

    Returns:
    beta_dict (dictionary): index, time, and corresponding beta list

    """
    
    print("beta_ji generation started")
    
    sorted_edges = sorted(G.edges())
    
    beta_dict = {}
    betap_dict = {}

    beta_dict[0] = {edge: beta0 if edge in edge_dict[0]['edge_list'] else 0 for edge in sorted_edges}
    betap_dict[0] = {edge: alpha_hosp*beta0 if edge in edge_dict[0]['edge_list'] else 0 for edge in sorted_edges}

    deltat = edge_dict[1]['time']-edge_dict[0]['time']
    
    num_average = int(24*deltat_kMC/deltat)
    
    cnt = 0
    
    average_arr = []
    
    beta_dict_cnt = 1
    
    for i in range(len(edge_dict)):
        
        print(len(edge_dict), i)
                        
        values = np.asarray([1 if edge in edge_dict[i]['edge_list'] else 0 for edge in sorted_edges])
                
        average_arr.append(values)
        cnt += 1
        
        if cnt == num_average:
            
            average = np.mean(average_arr, axis = 0)
                        
            print(beta_dict_cnt, beta0*np.mean(average))
            
            beta_dict[beta_dict_cnt] = {edge: beta0*av for (edge, av) in zip(sorted_edges,average)}
            betap_dict[beta_dict_cnt] = {edge: alpha_hosp*beta0*av for (edge, av) in zip(sorted_edges,average)}

            beta_dict_cnt += 1
            
            average_arr = []
            cnt = 0

    print("beta_ji generation ended")

    return beta_dict, betap_dict
    
if __name__ == "__main__":

#%%   
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

    H, node_attribute_dict_E2I, node_attribute_dict_I2R, \
    node_attribute_dict_H2R, node_attribute_dict_I2H, \
    node_attribute_dict_I2D, node_attribute_dict_H2D = spontaneousTransitions(G)
            
    # simulate dynamics
    deltat_kMC = 0.125 # unit: fraction of day
    
    #{edge: beta0 for edge in G.edges()}, {edge: alpha_hosp*beta0 for edge in G.edges()}

#%%   
    times, states = temporalNetworkEpidemics(G, H, IC, return_statuses, deltat = deltat_kMC, T = 50)

    print(states['H'])
#%%
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
    axes[1].plot(times, (np.asarray(states['D']))/len(G), label = 'Death', color = 'C6')
    axes[1].legend(loc = 0)

    axes[0].set_xlabel('time [days]')
    axes[0].set_ylabel('fraction')
    axes[1].set_xlabel('time [days]')
    axes[1].set_ylabel('fraction')
    #axes[0].set_ylim([0,0.2])
    #axes[1].set_ylim([0,0.03])
    plt.tight_layout()
    plt.legend()
    plt.show()
