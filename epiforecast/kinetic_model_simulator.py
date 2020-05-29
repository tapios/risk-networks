import numpy as np
import networkx as nx
from collections import defaultdict
from epiforecast.kinetic_model_helper import (
    KM_sigma, KM_gamma, KM_gamma_prime, KM_h, KM_d, KM_dp, KM_complement_indices
)
from EoN import Gillespie_simple_contagion


class KineticModel:
  def __init__(self,
               edges,
               node_identifiers,
               mean_contact_duration_network,
               transition_rates,
               transmission_rate,
               hospital_transmission_reduction):
    """
    A class to implement a Kinetic Monte-Carlo solver on a provided network.
    
    Args
    -----
    edges (np.array): a [num edges x 2] np.arrayof edges (corresponds to the
                    upper triangular of the adjacency matrix)
    node_identifiers (dict): a list of size 3, ["hospital_beds"] contains the node indices of the hospital beds
                                               ["health_workers"] contains the node indices of the health workers
                                               ["community"] contains the node indices of the community
    
    mean_contact_duration_network (np.array): The mean contact duration of each node in the static
                                              network over which we simulate
    transition_rates (TransitionRates): an object containing all the transition rates as dictionaries of size
                                      of health worker + community population
                                      for example: transition_rates.susceptible_to_exposed
    
    transmission_rate (float):  Global constant transmission rate (often referred within as beta)
    
    hospital_transmission_reduction (float): reduction factor for the transmission rate for those in hospital                                                   hospital_transmission_rate = transmission_rate *
                                                                        hospital_transmission_reduction
    """
    
    #Build networkx graph
    self.static_graph = nx.Graph() # a static graph with {0,1} edges
    
    self.static_graph.add_edges_from(edges)
    
    # independent rates diagram
    self.diagram_indep = nx.DiGraph()
    self.diagram_indep.add_node('P') # placeholder compartment (hosp. beds)
    self.diagram_indep.add_node('S')
    self.diagram_indep.add_edge('E', 'I', rate=1, weight_label='E->I')
    self.diagram_indep.add_edge('I', 'H', rate=1, weight_label='I->H')
    self.diagram_indep.add_edge('I', 'R', rate=1, weight_label='I->R')
    self.diagram_indep.add_edge('H', 'R', rate=1, weight_label='H->R')
    self.diagram_indep.add_edge('I', 'D', rate=1, weight_label='I->D')
    self.diagram_indep.add_edge('H', 'D', rate=1, weight_label='H->D')

    # neighbor-induced rates diagram
    self.diagram_neigh = nx.DiGraph()
    self.diagram_neigh.add_edge(
      ('I','S'), ('I','E'), rate=1, weight_label='SI->E'
    )
    self.diagram_neigh.add_edge(
      ('H','S'), ('H','E'), rate=1, weight_label='SH->E'
    )

    
    #set the transition rates:    
    nx.set_node_attributes(self.static_graph, values=transition_rates.exposed_to_infected, name='E->I')
    nx.set_node_attributes(self.static_graph, values=transition_rates.infected_to_hospitalized, name='I->H')
    nx.set_node_attributes(self.static_graph, values=transition_rates.infected_to_resistant, name='I->R')
    nx.set_node_attributes(self.static_graph, values=transition_rates.infected_to_deceased, name='I->D')
    nx.set_node_attributes(self.static_graph, values=transition_rates.hospitalized_to_resistant, name='H->R')
    nx.set_node_attributes(self.static_graph, values=transition_rates.hospitalized_to_deceased, name='H->D')

    #set the transmission rates
    self.edges=edges
    self.transmission_rate = transmission_rate
    self.hospital_transmission_rate = transmission_rate * hospital_transmission_reduction
    self.update_contacts(mean_contact_duration_network)

    # set the initial node statuses
    # hospital bed nodes
    self.__P0 = node_identifiers["hospital_beds"]
    self.__HCW = node_identifiers["health_workers"]
    
    # what statuses to return from Gillespie simulation
    self.return_statuses = ('S', 'E', 'I', 'H', 'R', 'D', 'P')

  def update_contacts(self, mean_contact_duration_network):
    community_network = {tuple(edge): self.transmission_rate * mean_contact_duration_network[edge[0], edge[1]]
                         for edge in self.edges}
    hospital_network  = {tuple(edge): self.hospital_transmission_rate * mean_contact_duration_network[edge[0], edge[1]]
                         for edge in self.edges}
    nx.set_edge_attributes(self.static_graph, values=community_network, name='SI->E')
    nx.set_edge_attributes(self.static_graph, values=hospital_network, name='SH->E')
    
  def simulate(self,
               node_statuses,
               static_contact_interval):
    
    res = Gillespie_simple_contagion(self.static_graph,
                                     self.diagram_indep,
                                     self.diagram_neigh,
                                     node_statuses,
                                     self.return_statuses,
                                     return_full_data = True,
                                     tmin = 0.0,
                                     tmax = static_contact_interval)
    
    times, states = res.summary()
    self.node_statuses = res.get_statuses(time=times[-1])
    
    self.vacate_placeholder() # remove from hospital whoever recovered/died
    self.populate_placeholder() # move into hospital those who need it
    
    return self.node_statuses #synthetic data
  
  def vacate_placeholder(self):
    '''
    Vacate placeholder nodes if their status is not 'H'
    '''
    for i in self.__P0:
      if self.node_statuses[i] != 'P' and self.node_statuses[i] != 'H':
        self.node_statuses[ self.P_taken_by[i] ] = self.node_statuses[i]
        self.node_statuses[i] = 'P'
        self.P_taken_by[i] = i

  def populate_placeholder(self):
    '''
    Put 'H' nodes currently outside 'P' into 'P' slots
    '''
    P_all_nodes = np.nonzero(self.node_statuses == 'P')[0]
    P_nodes = P_all_nodes[ P_all_nodes <= self.__P0[-1] ]

    if P_nodes.size != 0:
      H_all_nodes = np.nonzero(self.node_statuses == 'H')[0]
      H_nodes = H_all_nodes[ H_all_nodes > self.__P0[-1] ]
      for i in range(min(P_nodes.size, H_nodes.size)):
        self.node_statuses[ P_nodes[i] ] = 'H'
        self.node_statuses[ H_nodes[i] ] = 'P'
        self.P_taken_by[ P_nodes[i] ] = H_nodes[i]

