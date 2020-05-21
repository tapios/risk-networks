import numpy as np
from kMC_helper import *

class TemporalAdjacency:
  '''
  Piecewise-constant (in time) adjacency matrix with averaging functionality
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
    self.beta  = None # 2D np.array; (edges) x (number of time intervals)
    self.betap = None # 2D np.array; -----------------"-----------------

  def set_edge_list(self, edge_list):
    self.edge_list = edge_list

  def set_initial_active(self, initial_active):
    self.initial_active = initial_active

  def get_interval(self, t):
    '''
    Get the number of the averaging time interval from t
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
      t: days
    '''
    return max(
        lambda_min,
        lambda_max * (1 - np.cos(np.pi * t)**2)
    )

  def generate(self, dt_sync=0.004, muc=144, lambda_min=5/24, lambda_max=22/24):
    '''
    Generate times of activation/deactivation and piecewise-constant weights

    Defaults:
      dt_sync: 0.004 [days] or 5.76 [minutes]
      muc: 144 [1/day] or 6 [1/h]
      labmda_min:  5/24 [???]
      labmda_max: 22/24 [???]
    '''
    M = self.edge_list.shape[0] # total number of edges

    # sample indices of initially active edges uniformly at random
    active = np.random.choice(
        [False, True],
        size=M,
        p=[1 - self.initial_active, self.initial_active]
    )
    #inactive = kMC_complement_indices(M, active)

    #active_edges = self.edge_list[active]
    #inactive_edges = self.edge_list[inactive]

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

      # XXX maybe be sped up by generating random sequence outside loop
      if np.random.random() < Q1/(Q1+Q2):
        # sample uniformly at random which edge to deactivate
        k = np.random.choice(active_count)
        ind_to_move = np.where(active)[0][k] # piece of black magic, yum!
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
    # XXX to leave it, change (steps) to (steps+1) below
    self.times.resize(steps, refcheck=False)
    self.temporal_edge_list = np.copy(self.temporal_edge_list[:,:steps])

  def average_betas(self, dt_averaging=0.125, beta0=1.0, alpha_hosp=0.25):
    '''
    Average betas over generated times and weights

    Defaults:
      dt_averaging:     0.125 [days]
      beta0:      1.0  [???]
      alpha_hosp: 0.25 [???] fraction of beta0 for hospital nodes
    '''
    # XXX there's something funky going on: why do we sort here but not in
    # `generate`?
    #sorted_edges = np.copy(
    #    self.edge_list[ self.edge_list[:,0].argsort(kind='stable') ]
    #)
    #average_dt = np.mean(self.times[1:] - self.times[:-1])

    self.dt_averaging = dt_averaging
    kMC_timespan = np.arange(self.t0, self.t1, step=dt_averaging)
    if kMC_timespan[-1] < self.t1:
      kMC_timespan = np.append(kMC_timespan, self.t1)

    # using dt_averaging as a timestep, determine averaging intervals
    jump_indices = np.searchsorted(self.times, kMC_timespan)
    jump_indices[-1] += 1 # make sure the last column is included

    self.beta  = np.zeros( (self.edge_list.shape[0], jump_indices.size - 1) )
    self.betap = np.zeros( (self.edge_list.shape[0], jump_indices.size - 1) )
    for j in range(jump_indices.size - 1):
      chunk = self.temporal_edge_list[ :, jump_indices[j] : jump_indices[j+1] ]
      average = np.mean(chunk, axis=1)
      #print(j, np.mean(average))
      self.beta [:,j] = average
      self.betap[:,j] = average

    self.beta  *= beta0
    self.betap *= beta0 * alpha_hosp





