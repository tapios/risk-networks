import numpy as np

# age-dependent hospitalization and recovery rates
age_h  = np.array([0.020, 0.170, 0.250, 0.35, 0.45])
age_d  = np.array([0.001, 0.001, 0.005, 0.02, 0.05])
age_dp = np.array([0.001, 0.001, 0.010, 0.04, 0.10])

################################################################################
# helper functions #############################################################
################################################################################
def kMC_sigma(size):
  '''
  E->I
  Latent period
  '''
  return 2 + np.random.gamma(1.7, 2, size = size)

def kMC_gamma(size):
  '''
  (I->R) + (I->H) + (I->D)
  Duration of infectiousness for infected
  '''
  return 1 + np.random.gamma(1.5, 2, size = size)

def kMC_gamma_prime(size):
  '''
  (H->R) + (H->D)
  Duration of infectiousness for hospitalized
  '''
  return 1 + np.random.gamma(1.5, 3, size = size)

def kMC_h(array, beta = 4.0):
  '''
  I->H (component)
  '''
  return np.random.beta(beta * age_h[array] / (1 - age_h[array]), b = beta)

def kMC_d(array, beta = 4.0):
  '''
  I->D (component)
  '''
  return np.random.beta(beta * age_d[array] / (1 - age_d[array]), b = beta)

def kMC_dp(array, beta = 4.0):
  '''
  H->D (component)
  '''
  return np.random.beta(beta * age_dp[array] / (1 - age_dp[array]), b = beta)

def kMC_complement_indices(N, idx):
  '''
  Return a mask (boolean np.array) to select complement of idx

  Usage:
    N = 10
    nodes = np.zeros(N)
    HCW = np.array([5,6,7]) # indices
    not_HCW = kMC_reverse_indices(N, HCW)

    nodes[HCW] = 1
    nodes[not_HCW] = -1

  Note: the 'fancy indexing' mechanisms may be different;
  in the first case, HCW is an array of indices of length <= N,
  in the second case, not_HCW is a boolean array of length == N.

  This seems slightly wrong from design point of view, but OK for now.
  However, this function *must* return a mask; that's what other code expects.
  '''
  not_idx = np.ones(N, np.bool) # create a mask
  not_idx[idx] = False
  return not_idx

def kMC_print_states(t_offset, times, states, which):
  '''
  Print summary of states using output from networkx
  '''

  print(
      states['S'][-1],
      states['E'][-1],
      states['I'][-1],
      states['H'][-1],
      states['R'][-1],
      states['D'][-1]
  )

