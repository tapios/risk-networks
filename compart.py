import numpy as np
from matplotlib import pyplot

class Compartmental:
  """
  A simple class that implements compartmental models

  The class computes RHS's to make use of scipy's ODE solvers.

  Parameters:
    beta        transmission rate
    gamma       recovery rate
    a           inverse of the average incubation time

  We assume S + E + I + R + ... = 1, so eliminate number of individuals N.
  Whatever the model is, we do not compute the last compartment because it can
  be inferred from equation above.

  References:
    [1] Bertozzi et al.
        The challenges of modeling and forecasting thespread of COVID-19.
        https://arxiv.org/pdf/2004.04741.pdf
  """

  def __init__(_s, beta = 0.324, gamma = 0.12, a = 0.3):
    _s.beta = beta
    _s.gamma = gamma
    _s.a = a

  def SEIR(_s, t, z):
    """SEIR's model RHS
      z[0] = S
      z[1] = E
      z[2] = I
      R is inferred from
        S + E + I + R = const
    """
    rhs = np.empty(3)
    IS = z[2] * z[0]
    rhs[0] = - _s.beta * IS
    rhs[1] = + _s.beta * IS - _s.a * z[1]
    rhs[2] = + _s.a * z[1] - _s.gamma * z[2]

    return rhs

################################################################################
# end of Compartmental #########################################################
################################################################################
