import numpy as np
import scipy.sparse as scspa
from epiforecast.kinetic_model_helper import *

class TemporalAdjacency:
  '''
  Piecewise-constant (in time) adjacency matrix with generation and averaging
  '''

  def __init__(self, t0 = 0.0, t1 = 1.0):
    self.t0 = t0 # in days
    self.t1 = t1 # ---"---

    self.edge_list = None # 2D np.array; static, {0,1} edge weights
    self.temporal_edge_list = None # 2D boolean np.array; masks edge_list
    self.times = None # 1D np.array of times; dt isn't necessarily fixed
    # Note: temporal_edge_list.shape[1] == times.size

    # a fraction of initially active edges
    self.initial_active = 0.0

    self.dt_averaging = None # float; typically, larger than dt's in self.times
    self.wji  = None # 2D np.array; (edges) x (number of time intervals)
    self.wjip = None # 2D np.array; -----------------"-----------------

  def load_edge_list(self, filename):
    # XXX this also needs to be disentangled
    edge_list = np.loadtxt(filename, dtype=int, comments='#')

    # ensure edge_list only contains upper triangular edges (i.e. symmetric)
    upp_tr = edge_list[:,0] < edge_list[:,1]
    self.edge_list = edge_list[upp_tr]

  def set_initial_active(self, initial_active):
    '''
    Set a fraction (float) of active edges at the start of the interval
    '''
    self.initial_active = initial_active

  def get_interval_index(self, t):
    '''
    Infer the index of the averaging time interval from t
    '''
    index = 0
    if t >= self.t1:
      index = int((self.t1 - self.t0) // self.dt_averaging) - 1
    elif t >= self.t0:
      index = int((t - self.t0) // self.dt_averaging)
    else:
      index = -1
      raise ValueError('t < t0, ambiguous')

    return index

  def mean_contact_rate(self, t, lambda_min, lambda_max):
    '''
    Compute mean contact rate

    Units:
      t:           [day]
      lambda_min:  [1/day]
      lambda_max:  [1/day]
    '''
    return max(
        lambda_min,
        lambda_max * (1 - np.cos(np.pi * t)**4)**4
    )

  def generate(self, dt_sync=0.004, muc=1920, lambda_min=3, lambda_max=22):
    '''
    Generate times of activation/deactivation and piecewise-constant weights

    Defaults:
      dt_sync: 0.004 [days] or 5.76 [minutes]
      muc: 1920 [1/day] or 1/45 [1/s]
      labmda_min:  3 [1/day]
      labmda_max: 22 [1/day]
    '''
    M = self.edge_list.shape[0] # total number of edges

    # sample indices of initially active edges uniformly at random
    active = np.random.choice(
        [False, True],
        size=M,
        p=[1 - self.initial_active, self.initial_active]
    )
    #inactive = KM_complement_indices(M, active)

    t = 0.0
    t_next_sync = dt_sync
    steps = 0 # time steps generated
    mem_steps = 100 # memory allocated for this many time steps
    self.times = np.zeros(mem_steps)
    self.temporal_edge_list = np.zeros( (M, mem_steps), dtype=np.bool )

    self.times[0] = self.t0
    self.temporal_edge_list[:,0] = active

    while t < self.t1 - self.t0:
      active_count = np.count_nonzero(active)
      # rate process 1: deactivation
      Q1 = muc * active_count

      # rate process 3: activation
      Q2 = self.mean_contact_rate(t,lambda_min,lambda_max) * (M - active_count)

      # XXX may be sped up by generating random sequence outside loop
      if np.random.random() < Q1/(Q1+Q2):
        # sample uniformly at random which edge to deactivate
        k = np.random.choice(active_count)
        ind_to_move = np.where(active)[0][k] # piece of black magic, woohoo!
        active[ind_to_move] = False
      else:
        # sample uniformly at random which edge to activate
        k = np.random.choice(M - active_count)
        ind_to_move = np.where(~active)[0][k]
        active[ind_to_move] = True

      # XXX same: possible speed-ups
      t += -np.log(np.random.random()) / (Q1+Q2)

      if t >= t_next_sync:
        t_next_sync += dt_sync
        steps += 1

        # memory allocation
        if steps >= mem_steps:
          mem_steps = steps * 2
          self.times.resize(mem_steps)

          # resize does not work for 2D arrays (scrambles values)
          new_edge_list = np.zeros( (M, mem_steps), dtype=np.bool )
          new_edge_list[:, :steps] = np.copy(self.temporal_edge_list)
          self.temporal_edge_list = new_edge_list

        self.times[steps] = t
        self.temporal_edge_list[:,steps] = active

    # trim trailing zeros;
    # XXX also, shave off one last step because it's > 1.0
    # XXX to leave it, change (steps) to (steps+1) in two lines below
    self.times.resize(steps, refcheck=False)
    self.temporal_edge_list = np.copy(self.temporal_edge_list[:,:steps])

  def average_wjis(self, dt_averaging=0.125):
    '''
    Average w_{ji}'s over generated times and weights

    Defaults:
      dt_averaging: 0.125 [day]
    '''
    self.dt_averaging = dt_averaging
    KM_timespan = np.arange(self.t0, self.t1, step=dt_averaging)
    if KM_timespan[-1] < self.t1:
      KM_timespan = np.append(KM_timespan, self.t1)

    # using dt_averaging as a timestep, determine averaging intervals
    jump_indices = np.searchsorted(self.times, KM_timespan)
    jump_indices[-1] += 1 # make sure the last column is included

    self.wji  = np.zeros( (self.edge_list.shape[0], jump_indices.size - 1) )
    self.wjip = np.zeros( (self.edge_list.shape[0], jump_indices.size - 1) )
    for j in range(jump_indices.size - 1):
      chunk = self.temporal_edge_list[ :, jump_indices[j] : jump_indices[j+1] ]
      average = np.mean(chunk, axis=1)
      #print(j, np.mean(average))
      self.wji [:,j] = average
      self.wjip[:,j] = average

  def multiply_wjis(self, factor, factor_p):
    self.wji  *= factor
    self.wjip *= factor_p

  def get_wjis(self, t, structure, shape=None):
    '''
    Get wji and wji^prime at time t using self.wji, self.wjip

    Args:
      t [day]: time at which to get wji, wjip
      structure: which data structure to use for output
      shape: only used for 'sparse' type of structure
    '''
    # find which time interval we are currently in
    j = self.get_interval_index(t)

    if structure == 'dict':
      # get the info from wji, wjip into dictionaries (required by networkx)
      wji_output  = {}
      wjip_output = {}
      for k,e in enumerate(self.edge_list):
        wji_output [tuple(e)] = self.wji [k,j]
        wjip_output[tuple(e)] = self.wjip[k,j]
    elif structure == 'sparse':
      wji_output = scspa.csr_matrix(
          (self.wji[:,j], (self.edge_list[:,0], self.edge_list[:,1])), shape
      )
      wjip_output = scspa.csr_matrix(
          (self.wjip[:,j], (self.edge_list[:,0], self.edge_list[:,1])), shape
      )
      # symmetrize (because edge_list is only upper triangular)
      wji_output  += wji_output.T
      wjip_output += wjip_output.T

    return wji_output, wjip_output



