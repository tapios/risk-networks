import numpy as np
import networkx as nx
from collections import defaultdict
from epiforecast.kinetic_model_helper import (
    KM_sigma, KM_gamma, KM_gamma_prime, KM_h, KM_d, KM_dp, KM_complement_indices
)
from EoN import Gillespie_simple_contagion

class KineticModel:
  # these are just for reference; not actually used in code
  indep_weight_labels = [
      'E->I', # Latent period
      'I->R', # Duration of infectiousness for infected
      'H->R', # Duration of infectiousness for hospitalized
      'I->H', # Hospitalization rate
      'I->D', # Death rate for infected
      'H->D', # Death rate for hospitalized
  ]
  neigh_weight_labels = [
      'SI->E', # beta
      'SH->E'  # beta^prime
  ]

  def __init__(self, edge_list):
    self.static_graph = nx.Graph() # a static graph with {0,1} edges
    self.static_graph.add_edges_from(edge_list)


    # independent rates diagram
    self.diagram_indep = nx.DiGraph()
    self.diagram_indep.add_node('P') # placeholder compartment (hosp. beds)
    self.diagram_indep.add_node('S')
    self.diagram_indep.add_edge('E', 'I', rate=1, weight_label='E->I')
    self.diagram_indep.add_edge('I', 'R', rate=1, weight_label='I->R')
    self.diagram_indep.add_edge('H', 'R', rate=1, weight_label='H->R')
    self.diagram_indep.add_edge('I', 'H', rate=1, weight_label='I->H')
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

    self.IC = None # initial conditions for Gillespie
    self.__P0  = None # array of initial placeholder nodes
    self.__HCW = None # array of healthcare-worker nodes
    self.__I0  = None # array of initially infected nodes
    self.P_taken_by = None # array that maps which P nodes were taken by which

    # numpy arrays (when initialized) of independent rates
    self.sigma  = None # sigma_i
    self.gamma  = None # gamma_i
    self.gammap = None # gamma^prime_i
    self.h      = None # h_i
    self.d      = None # d_i
    self.dp     = None # d^prime_i

    # age distribution: a numpy array that sums to 1, and an index array
    self.ages = None
    self.working = None

    # what statuses to return from Gillespie simulation
    self.return_statuses = ('S', 'E', 'I', 'H', 'R', 'D', 'P')

  def get_IC(self):
    return (self.__P0, self.__HCW, self.__I0)

  def set_return_statuses(self, which='all'):
    '''
    Set which statuses to return from Gillespie simulation

    The parameter which can be:
      - 'all':  all statuses are reported
      - 'noP':  all except placeholder
      - string: specify explicitly, like 'SIR' or 'SIRD' etc.
    '''
    if which == 'all':
      self.return_statuses = ('S', 'E', 'I', 'H', 'R', 'D', 'P')
    elif which == 'noP':
      self.return_statuses = ('S', 'E', 'I', 'H', 'R', 'D')
    else:
      self.return_statuses = tuple(which)

  def set_ages(
      self,
      ages=[0.2298112587,0.3876994201,0.2504385036,0.079450985,0.0525998326],
      working=[1,2]
      ):
    self.ages = np.array(ages)
    self.working = np.array(working)

  def set_IC(self, P0=0.005, HCW=0.045, I0=0.01):
    '''
    Set initial placeholder, infected and healthcare-worker nodes

    P0:       (float)
              a fraction of nodes that starts from 0
    HCW:      (array_like of ints, or float)
              an array of nodes or a fraction of nodes that starts after P0
    I0:       (array_like of ints, or float)
              an array of nodes or a fraction of nodes to sample infected
    '''
    N = self.static_graph.number_of_nodes()
    self.IC = np.empty(N, dtype='<U1') # numpy array of characters
    self.IC.fill('S') # initialize as susceptible

    self.__P0 = np.arange(int(P0 * N)) # essentially, immutable
    self.P_taken_by = np.arange(int(P0 * N)) # mutable

    if type(HCW) == float:
      self.__HCW = np.arange(int(HCW * N)) + int(P0 * N)
    else:
      self.__HCW = np.array(HCW)

    if type(I0) == float:
      p = np.ones(N) / (N - self.__P0.size)
      p[self.__P0] = 0 # avoid sampling placeholder nodes
      self.__I0 = np.random.choice(N, size=int(I0 * N), replace=False, p=p)
    else:
      self.__I0 = np.array(I0)

    # check for interesections
    P0_I0  = np.in1d(self.__P0, self.__I0)
    P0_HCW = np.in1d(self.__P0, self.__HCW)
    if P0_I0.any():
      print("WARNING: P0 intersect I0 != 0, some of I0 will be overwritten")
      mask = np.searchsorted(self.__I0, self.__I0[P0_I0])
      self.__I0 = np.delete(self.__I0, mask)
    if P0_HCW.any() > 0:
      print("WARNING: P0 intersect HCW != 0, some of HCW will be overwritten")
      mask = np.searchsorted(self.__HCW, self.__HCW[P0_HCW])
      self.__HCW = np.delete(self.__HCW, mask)

    self.IC[ self.__I0 ] = 'I'
    self.IC[ self.__P0 ] = 'P'

  def set_independent_rates(self):
    N = self.static_graph.number_of_nodes()

    # sigma, gamma, gamma^prime don't depend on age; sample away
    self.sigma  = 1 / KM_sigma(N)
    self.gamma  = 1 / KM_gamma(N)
    self.gammap = 1 / KM_gamma_prime(N)

    # prepare arrays
    self.h  = np.zeros(N)
    self.d  = np.zeros(N)
    self.dp = np.zeros(N)

    # HCW are only working people, so sample only from those classes
    HCW_p = self.ages[self.working]
    HCW_p /= np.linalg.norm(HCW_p, 1)
    HCW_classes = np.random.choice(self.working, p=HCW_p, size=self.__HCW.size)

    # sample the rest: N - HCW; we do sample placeholders for simplicity
    rest_size = N - self.__HCW.size
    rest_classes = np.random.choice(self.ages.size, p=self.ages, size=rest_size)

    # age-dependent rates
    HCW_h  = KM_h (HCW_classes)
    HCW_d  = KM_d (HCW_classes)
    HCW_dp = KM_dp(HCW_classes)

    rest_h  = KM_h (rest_classes)
    rest_d  = KM_d (rest_classes)
    rest_dp = KM_dp(rest_classes)

    # combine HCW + rest into self arrays
    not_HCW = KM_complement_indices(N, self.__HCW) # complement of __HCW

    self.h [self.__HCW] = HCW_h
    self.d [self.__HCW] = HCW_d
    self.dp[self.__HCW] = HCW_dp
    self.h [not_HCW] = rest_h
    self.d [not_HCW] = rest_d
    self.dp[not_HCW] = rest_dp

    # rates construction done; now assign them to the graph edges
    EI_dict = dict(enumerate(self.sigma))
    IR_dict = dict(enumerate((1 - self.h - self.d) * self.gamma))
    IH_dict = dict(enumerate(self.h * self.gamma))
    ID_dict = dict(enumerate(self.d * self.gamma))
    HR_dict = dict(enumerate((1 - self.dp) * self.gammap))
    HD_dict = dict(enumerate(self.dp * self.gammap))

    nx.set_node_attributes(self.static_graph, values=EI_dict, name='E->I')
    nx.set_node_attributes(self.static_graph, values=IR_dict, name='I->R')
    nx.set_node_attributes(self.static_graph, values=IH_dict, name='I->H')
    nx.set_node_attributes(self.static_graph, values=ID_dict, name='I->D')
    nx.set_node_attributes(self.static_graph, values=HR_dict, name='H->R')
    nx.set_node_attributes(self.static_graph, values=HD_dict, name='H->D')

  def update_beta_rates(self, beta_dict, betap_dict):
    '''
    Update beta and beta^prime
    '''
    nx.set_edge_attributes(self.static_graph, values=beta_dict,  name='SI->E')
    nx.set_edge_attributes(self.static_graph, values=betap_dict, name='SH->E')

  def do_Gillespie_step(self, t, dt):
    res = Gillespie_simple_contagion(
        self.static_graph,
        self.diagram_indep,
        self.diagram_neigh,
        self.IC,
        self.return_statuses,
        return_full_data = True,
        tmin = t,
        tmax = t+dt
    )
    return res

  def update_IC(self, node_status):
    # node_status is a dict, so have to convert to a numpy array
    self.IC[ list(node_status.keys()) ] = list(node_status.values())

  def vacate_placeholder(self):
    '''
    Vacate placeholder nodes if their status is not 'H'
    '''
    for i in self.__P0:
      if self.IC[i] != 'P' and self.IC[i] != 'H':
        self.IC[ self.P_taken_by[i] ] = self.IC[i]
        self.IC[i] = 'P'
        self.P_taken_by[i] = i

  def populate_placeholder(self):
    '''
    Put 'H' nodes currently outside 'P' into 'P' slots
    '''
    P_all_nodes = np.nonzero(self.IC == 'P')[0]
    P_nodes = P_all_nodes[ P_all_nodes <= self.__P0[-1] ]

    if P_nodes.size != 0:
      H_all_nodes = np.nonzero(self.IC == 'H')[0]
      H_nodes = H_all_nodes[ H_all_nodes > self.__P0[-1] ]
      for i in range(min(P_nodes.size, H_nodes.size)):
        self.IC[ P_nodes[i] ] = 'H'
        self.IC[ H_nodes[i] ] = 'P'
        self.P_taken_by[ P_nodes[i] ] = H_nodes[i]

# end of class KineticModel


