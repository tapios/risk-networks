import os
import numpy as np
import networkx as nx
from collections import defaultdict
from TemporalAdjacency import TemporalAdjacency
from KM_helper import *
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

  def __init__(self):
    self.static_graph = nx.DiGraph() # a static graph with {0,1} edges
    self.fallback_edges_filename = os.path.join(
        '..', 'data', 'networks', 'edge_list_SBM_1e3.txt'
    )

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

    self.IC = defaultdict(lambda: 'S') # initial conditions for Gillespie
    self.__P0  = None # an array of initial placeholder nodes
    self.__HCW = None # an array of healthcare-worker nodes
    self.__I0  = None # an array of initially infected nodes

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

    # an object that handles everything time-dependency-related
    self.TA = TemporalAdjacency()

    # what statuses to return from Gillespie simulation
    self.return_statuses = ('S', 'E', 'I', 'H', 'R', 'D', 'P')

  def get_IC(self):
    return (self.__P0, self.__HCW, self.__I0)

  def load_edge_list(self, filename=None):
    if filename is None:
      filename = self.fallback_edges_filename

    edge_list = np.loadtxt(filename, dtype=int, comments='#')
    self.static_graph.add_edges_from(edge_list)
    self.TA.set_edge_list(edge_list)

  def set_statuses(self, which='all'):
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

  def set_IC(self, P0=0.005, HCW=0.045, I0=0.01, active=0.034):
    '''
    Set initial placeholder, infected and healthcare-worker nodes

    P0:       (array_like of ints, or float)
              an array of nodes or a fraction of nodes that starts from 0
    HCW:      (array_like of ints, or float)
              an array of nodes or a fraction of nodes that starts after P0
    I0:       (array_like of ints, or float)
              an array of nodes or a fraction of nodes to sample infected
    active:   (float)
              a fraction of active edges at the start of simulation;
    '''
    self.TA.set_initial_active(active)

    N = self.static_graph.number_of_nodes()

    if type(P0) == float:
      self.__P0 = np.arange(int(P0 * N))
    else:
      self.__P0 = np.array(P0)

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

    for i in self.__I0:
      self.IC[i] = 'I'

    for i in self.__P0:
      self.IC[i] = 'P'

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

  def generate_temporal_adjacency(self, **kwargs):
    self.TA.generate(**kwargs)

  def average_betas(self, **kwargs):
    self.TA.average_betas(**kwargs)

  def update_beta_rates(self, t):
    '''
    Update beta and beta^prime at time t using self.TA
    '''
    # find which time interval we are currently in
    j = self.TA.get_interval(t)

    # get the info from self.TA into dictionaries (required by networkx)
    beta_dict  = {}
    betap_dict = {}
    for k,e in enumerate(self.TA.edge_list):
      beta_dict [tuple(e)] = self.TA.beta [k,j]
      betap_dict[tuple(e)] = self.TA.betap[k,j]

    # update the rates
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


# end of class KineticModel


