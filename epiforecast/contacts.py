import numpy as np
import scipy.sparse as scspa
from epiforecast.kinetic_model_helper import *

def load_edges(filename):
    """
    Return a list of unique edges

    Args
    ----
    filename (str): path to text file with a list of edges
    """
    edges = np.loadtxt(filename, dtype=int, comments='#')

    # Remove non-unique edges, when they are included in `filename`
    unique_edges = edges[:, 0] < edges[:, 1]

    return edges[unique_edges]




class StaticNetworkTimeSeries:
    """
    Container for a time series of networks.
    """
    def __init__(self, edges):
        """
        edges (np.array): An array of size `number_of_edges x 2` giving the `i, j` coordinates
                          of each active edge.
        """

        self.edges = edges
        self.nodes = np.max(self.edges) + 1
        self.contact_networks = []
        self.time = []
      
    def add_network(self, average_contact_duration, time):
        """
        Args
        ----

        average_contact_duration (np.array): An array of size `number_of_edges` giving the
                                             average contact duration of each edge in `edges`
                                             on a time interval starting at `time`.

        time (number): the time of day to which average_contact_duration pertains.
        """

        self.time.append(time)
        
        # From the scspa.csr_matrix docstring:
        #
        # csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
        #
        #       where ``data``, ``row_ind`` and ``col_ind`` satisfy the
        #       relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

        # This generates only the upper triangular part of the contact network:

        contact_network = scspa.csr_matrix((average_contact_duration, (self.edges[:, 0], self.edges[:, 1])),
                                           shape=(self.nodes, self.nodes))

        self.contact_networks.append(contact_network + contact_network.T)

    def add_networks(self, average_contact_durations, times=None):

        if times is None:
            times = np.arange(0.0, stop=1.0, step=1.0/average_contact_durations.shape[1])

        for i, time in enumerate(times):
            self.add_network(average_contact_durations[:, i], time)




class ContactGenerator:
    '''
    Class for generating time-averaging contact networks.
    '''

    def __init__(self, edges, initial_active, mu, t0 = 0.0, t1 = 1.0):
        '''
        Constructor

        Args:
            initial_active [1]: a fraction [0, 1] of active edges at t0
            mu [1/day]: (mean contact duration)**(-1)
            t0 [day]: start of the interval
            t1 [day]: end   of the interval
        '''
        self.initial_active = initial_active
        self.mu = mu
        self.t0 = t0
        self.t1 = t1

        self.edges = edges
        self.temporal_edges = None # 2D boolean np.array; masks edges
        self.times = None # 1D np.array of times; dt isn't necessarily fixed
        # Note: temporal_edges.shape[1] == times.size

        self.averaging_interval = None # float; typically, larger than dt's in self.times
        self.wji  = None # 2D np.array; (edges) x (number of time intervals)

    def get_interval_index(self, t):
        '''
        Infer the index of the averaging time interval from t
        '''
        index = 0
        if t >= self.t1:
            index = int((self.t1 - self.t0) // self.averaging_interval) - 1
        elif t >= self.t0:
            index = int((t - self.t0) // self.averaging_interval)
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
        return max(lambda_min,
                   lambda_max * (1 - np.cos(np.pi * t)**4)**4
                  )

    def generate_static_networks(self, averaging_interval = 0.125,
                                                  dt_sync = 0.004,
                                               lambda_min = 3,
                                               lambda_max = 22):
        '''
        Generate times of activation/deactivation and piecewise-constant weights, then 
        average w_{ji}'s over generated times and weights

        Args
        ----

        static_network_list (StaticNetworkTimeSeries): Container for the networks

        averaging_interval  (float): length of the static network interval (i.e length of averages)

        dt_sync             (float):

        lambda_min          (float): minimum mean contact rate 

        lambda_max          (float): maximum mean contact rate

        Defaults
        --------
        averaging_interval: 0.125 [day]

        dt_sync: 0.004 [days] or 5.76 [minutes]

        lambda_min:  3 [1/day]

        lambda_max: 22 [1/day]
        '''    

        self.simulate_temporal_contacts(dt_sync, lambda_min, lambda_max)
        self._average_wjis(averaging_interval)

        # timeseries.add_networks(contacts_generator.generate_static_networks())
        return self.wji
    
    def simulate_temporal_contacts(self, dt_sync, lambda_min, lambda_max):
        """
        Simulate time-dependent contacts with a birth/death process.
        """
        total_edges = self.edges.shape[0]
  
        # Uniform random sample of edges that are initially active
        active = np.random.choice([False, True],
                                  size = total_edges,
                                  p = [1 - self.initial_active, self.initial_active])
  
        t = 0.0
        t_next_sync = dt_sync
        steps = 0 # time steps generated
        mem_steps = 100 # memory allocated for this many time steps
        self.times = np.zeros(mem_steps)
        self.temporal_edges = np.zeros((total_edges, mem_steps), dtype=np.bool)
  
        self.times[0] = self.t0
        self.temporal_edges[:, 0] = active

        while t < self.t1 - self.t0:
            active_count = np.count_nonzero(active)
      
            deactivation_rate = self.mu * active_count

            activation_rate = self.mean_contact_rate(t, lambda_min, lambda_max) * (total_edges - active_count)
  
            deactivation_probability = deactivation_rate / (deactivation_rate + activation_rate)
  
            # NOTE: may be sped up by generating random sequence outside loop
            #
            # Draw from uniform random distribution on [0, 1) to decide
            # whether to activate or deactivate edges
            if np.random.random() < deactivation_probability: # deactivate edges
  
                k = np.random.choice(active_count)
                ind_to_move = np.where(active)[0][k] # piece of black magic, woohoo!
                active[ind_to_move] = False
  
            else: # activate edges
  
                k = np.random.choice(total_edges - active_count)
                ind_to_move = np.where(~active)[0][k]
                active[ind_to_move] = True
      
            # NOTE: possible speed-ups (?)
            t += -np.log(np.random.random()) / (deactivation_rate + activation_rate)
      
            if t >= t_next_sync:
                t_next_sync += dt_sync
                steps += 1
      
                # Memory allocation
                if steps >= mem_steps:
                    mem_steps = steps * 2
                    self.times.resize(mem_steps)
      
                    # Resize does not work for 2D arrays (scrambles values)
                    new_edges = np.zeros((total_edges, mem_steps), dtype=np.bool)
                    new_edges[:, :steps] = np.copy(self.temporal_edges)
                    self.temporal_edges = new_edges
      
                self.times[steps] = t
                self.temporal_edges[:, steps] = active
      
        # Trim trailing zeros;
        # NOTE also, shave off one last step because it's > 1.0
        # NOTE to leave it, change (steps) to (steps+1) in two lines below
        self.times.resize(steps, refcheck=False)
        self.temporal_edges = np.copy(self.temporal_edges[:, :steps])

    def _average_wjis(self, averaging_interval):
        """
        Average temporally-evolving the w_{ji}'s over the `averaging_interval`
        """
        self.averaging_interval = averaging_interval

        averaging_times = np.arange(self.t0, self.t1, step=averaging_interval)
  
        if averaging_times[-1] < self.t1:
            averaging_times = np.append(averaging_times, self.t1)
  
        # Using averaging_interval as a timestep, determine averaging intervals
        print(self.times)
        jump_indices = np.searchsorted(self.times, averaging_times)
        jump_indices[-1] += 1 # make sure the last column is included
  
        for j in range(jump_indices.size - 1):
           chunk = self.temporal_edges[:, jump_indices[j] : jump_indices[j + 1]]
           average = np.mean(chunk, axis=1)

           print(average)
  
           self.wji = np.zeros((self.edges.shape[0], jump_indices.size - 1))
           self.wji[:, j] = average

        print(self.wji[1, :])
