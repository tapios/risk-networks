import EoN
import sys
sys.path.append('../risknet')
import epimodels
import numpy as np

class MasterEqn:

    def __init__(self, G):

        N = len(list(G.nodes))
        self.epistatic = epimodels.static(G, N)

    def solve(self, params, state0, T, dt_max, t_range):
        self.epistatic.init(beta = np.exp(params[0]))
        self.epistatic.set_solver(T = T, dt = dt_max)
        ke = self.epistatic.solve(state0, t_range)
        return ke.y
