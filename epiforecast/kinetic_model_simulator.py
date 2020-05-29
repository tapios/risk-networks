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
               mean_contact_duration,
               transition_rates,
               community_transmission_rate,
               hospital_transmission_reduction):
    """
    A class to implement a Kinetic Monte-Carlo solver on a provided network.
    
    Args
    -----
    edges (np.array): a [num edges x 2] np.arrayof edges (corresponds to the
                    upper triangular of the adjacency matrix)
    node_identifiers (dict[np.array]): a dict of size 3, ["hospital_beds"] contains the node indices of the hospital beds
                                                         ["health_workers"] contains the node indices of the health workers
                                                         ["community"] contains the node indices of the community
    
    mean_contact_duration (np.array): The mean contact duration of each node in the static
                                      network over which we simulate
    transition_rates (TransitionRates): an object containing all the transition rates as dictionaries of size
                                      of health worker + community population
                                      for example: transition_rates.susceptible_to_exposed
    
    community_transmission_rate (float):  Global constant transmission rate for the community(often referred within as beta)
    
    hospital_transmission_reduction (float): reduction factor for the transmission rate for those in hospital
                                             hospital_transmission_rate = community transmission_rate *
                                                                          hospital_transmission_reduction
    """
    
    #Build networkx graph
    self.static_graph = nx.Graph() # a static graph with {0,1} edges
    self.static_graph.add_edges_from(edges)
    
    # independent rates diagram
    self.diagram_indep = nx.DiGraph()
    self.diagram_indep.add_node('P') # placeholder compartment (hosp. beds)
    self.diagram_indep.add_node('S')
    self.diagram_indep.add_edge('E', 'I', rate=1, weight_label='exposed_to_infected')
    self.diagram_indep.add_edge('I', 'H', rate=1, weight_label='infected_to_hospitalized')
    self.diagram_indep.add_edge('I', 'R', rate=1, weight_label='infected_to_resistant')
    self.diagram_indep.add_edge('H', 'R', rate=1, weight_label='hospitalized_to_resistant')
    self.diagram_indep.add_edge('I', 'D', rate=1, weight_label='infected_to_deceased')
    self.diagram_indep.add_edge('H', 'D', rate=1, weight_label='hospitalized_to_deceased')

    # neighbor-induced rates diagram
    self.diagram_neigh = nx.DiGraph()
    self.diagram_neigh.add_edge(
      ('I','S'), ('I','E'), rate=1, weight_label='SI->E'
    )
    self.diagram_neigh.add_edge(
      ('H','S'), ('H','E'), rate=1, weight_label='SH->E'
    )

    
    #set the transition rates:    
    nx.set_node_attributes(self.static_graph, values=transition_rates.exposed_to_infected, name='exposed_to_infected')
    nx.set_node_attributes(self.static_graph, values=transition_rates.infected_to_hospitalized, name='infected_to_hospitalized')
    nx.set_node_attributes(self.static_graph, values=transition_rates.infected_to_resistant, name='infected_to_resistant')
    nx.set_node_attributes(self.static_graph, values=transition_rates.infected_to_deceased, name='infected_to_deceased')
    nx.set_node_attributes(self.static_graph, values=transition_rates.hospitalized_to_resistant, name='hospitalized_to_resistant')
    nx.set_node_attributes(self.static_graph, values=transition_rates.hospitalized_to_deceased, name='hospitalized_to_deceased')

    #set the transmission rates
    self.edges=edges
    self.community_transmission_rate = community_transmission_rate
    self.hospital_transmission_rate = community_transmission_rate * hospital_transmission_reduction
    self.set_contact_network(mean_contact_duration)

    # keep node identifiers 
    self.node_identifiers=node_identifiers

    #will store the address of a patient in a hospital bed, by calling address_of_patient[hospital_bed]
    self.address_of_patient = np.repeat(-1,node_identifiers["hospital_beds"].size)

    # what statuses to return from Gillespie simulation
    self.return_statuses = ('S', 'E', 'I', 'H', 'R', 'D', 'P')

  def set_contact_network(self, mean_contact_duration):
    """
    Set the graph edge weights by multiplying the (community and hospital) transmission rates by the inputted `mean_contact_duration` and saving in a dict for
    networkx

    Args
    ----

    mean_contact_duration (np.array) : np.array of with [i,j] representing the edge between node i and j, holds the
                                       mean duration that the edge was 'active' during an averaging window 
    """
    community_network = {tuple(edge): self.community_transmission_rate * mean_contact_duration[edge[0], edge[1]]
                         for edge in self.edges}
    hospital_network  = {tuple(edge): self.hospital_transmission_rate * mean_contact_duration[edge[0], edge[1]]
                         for edge in self.edges}
    nx.set_edge_attributes(self.static_graph, values=community_network, name='SI->E')
    nx.set_edge_attributes(self.static_graph, values=hospital_network, name='SH->E')
    
  def simulate(self,
               node_statuses,
               static_contact_interval):
    """
    Performs the Gillespie solver with our given graph with current contact network

    Args
    ----

    node_statuses (dict) : a dict of {node number : node status} ; the initial condition for the solve step

    static_contact_interval (float) : the integration time (over a static contact network)
    
    """
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
    
    self.vacate_hospital_bed() # remove from hospital whoever recovered/died
    self.populate_hospital_bed() # move into hospital those who need it
    
    return self.node_statuses #synthetic data
  
  def vacate_hospital_bed(self):
    '''
    Vacate hospital_bed nodes if their status is not 'H'
    '''
    #check each hospital bed 
    for hospital_bed in self.node_identifiers["hospital_beds"]:
      #If there is a node occupying  the bed  then the value != 'P'
      #If the node is longer in hospitalized state then the value != 'H'
      if self.node_statuses[hospital_bed] != 'P' and self.node_statuses[hospital_bed] != 'H':
        #then move them back to their original nodal position
        self.node_statuses[self.address_of_patient[hospital_bed]] = self.node_statuses[hospital_bed]
        #and set the state of the bed back to unoccupied: 'P'
        print("sending home patient ",hospital_bed," to ",self.address_of_patient[hospital_bed], " in state ", self.node_status[hospital_bed])
        self.node_statuses[hospital_bed] = 'P'
        self.address_of_patient[hospital_bed] = -1 #an unattainable value may be useful for debugging
              
  def populate_hospital_bed(self):
    '''
    Put 'H' nodes currently outside hospital beds (`hospital_seeking`), into an unoccupied hospital bed (`new_patient`).
    Record where the patient came from (in `self.patient_home`) place a 'P' in it's network position.
    '''
    #check each hospital bed
    for hospital_bed in self.node_identifiers["hospital_beds"]:
      #if any bed is unoccupied, then value == 'P'
      if self.node_statuses[hospital_bed] == 'P':
        
        #obtain the nodes seeking to be hospitalized (state == 'H') 
        populace = np.hstack([self.node_identifiers["health_workers"] , self.node_identifiers["community"]])
  
        hospital_seeking=[i for i in populace if self.node_statuses[i]== 'H']
        if (len(hospital_seeking)>0):
          new_patient_address = hospital_seeking[0]      
          #move a patient into the hospital bed, keeping track of its address
          self.node_statuses[hospital_bed] = 'H'
          self.node_statuses[new_patient_address] = 'P'
          self.address_of_patient[hospital_bed]= new_patient_address
          print("receiving new patient from",new_patient_address," into bed ",hospital_bed)
       
  
