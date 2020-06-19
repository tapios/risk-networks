import numpy as np
import networkx as nx

class Intervention:
  """
    Store intervention strategy and compute interventions from state

    Currently can only do a simple strategy with thresholds for E(xposed) and
    I(infected) in a "binary or" fashion:
        apply intervention to node[i] if (E[i] > E_thr) or (I[i] > I_thr)

    The class is aware of ensemble members, i.e. E[i] and I[i] in the above are
    computed as ensemble means of respective E[i]'s and I[i]'s

    Methods:
        find_sick

    Example:
        network = ContactNetwork(edges_filename, identifiers_filename)
        N = network.get_count_node()
        M = ensemble_size
        intervention = Intervention(N, M, compartment_index, E_thr=0.7, I_thr=0.5)

        for k in range(time_steps):
            # kinetic, master, DA
            sick_nodes = intervention.find_sick(ensemble_states)
            network.isolate(sick_nodes)
  """

  def __init__(self, N, M, compartment_index, E_thr=0.5, I_thr=0.5):
    """
      Constructor

      Args:
        N:                    number of nodes
        M:                    number of ensemble members
        compartment_index:    dictionary mapping letters to indices
        E_thr:                threshold of the E compartment
        I_thr:                threshold of the I compartment
    """

    self.N = N
    self.M = M

    E = compartment_index['E']
    I = compartment_index['I']

    if E == -1: # reduced model, need to compute E from the rest
      self.E_slice = None
    else:
      self.E_slice = np.s_[E * N : (E+1) * N]

    if I == -1:
      self.E_slice = None
    else:
      self.I_slice = np.s_[I * N : (I+1) * N]

    assert 0 < E_thr < 1
    assert 0 < I_thr < 1
    self.E_thr = E_thr
    self.I_thr = I_thr

  def __get_complement_substate(self, ensemble_states):
    """
      Get the complement substate for the whole ensemble

      Args:
        ensemble_states: (M,c*N) np.array, where c is the number of compartments

      Example:
        if the model is reduced and one of the states is implicitly computed
        (say, E), then we obtain it by subtracting from 1:
          1 - (S + I + H + R + D)
    """

    return 1 - ensemble_states.reshape( (self.M, -1, self.N) ).sum(axis=1)

  def find_sick(self, ensemble_states):
    """
      Find node indices that are considered sick according to E and I thresholds

      Args:
        ensemble_states: (M,c*N) np.array, where c is the number of compartments
    """

    # both substates are (M, N) in shape
    if self.E_slice is None:
      E_substate = self.__get_complement_substate(ensemble_states)
    else:
      E_substate = ensemble_states[:, self.E_slice]

    if self.I_slice is None:
      I_substate = self.get_complement_substate(ensemble_states)
    else:
      I_substate = ensemble_states[:, self.I_slice]

    # both means are (N,) in shape
    E_ensemble_mean = E_substate.mean(axis=0)
    I_ensemble_mean = I_substate.mean(axis=0)

    return np.where(
        (E_ensemble_mean > self.E_thr) | (I_ensemble_mean > self.I_thr)
        )[0]


