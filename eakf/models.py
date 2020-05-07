import EoN
import sys
sys.path.append('../risknet')
import epimodels
import numpy as np

class MasterEqn:
    '''
    A class describing the time trajectory of an epidemic, which
    consists of 6 states by N nodes.
    '''

    def __init__(self, G, model = None):

        if model is None:
            self.model = epimodels.static(G, len(list(G.nodes)))
        else:
            self.model = model

    def solve(self, params, state0, T, dt_max, t_range):
        '''
         params (np.array) : The list of parameters to be inferred, such as (?)
                                - probability of infection / transmission rate (beta_ij)
                                - latent period (1 / sigma_i)
                                - duration of infectiousness in and out of hospitals (gamma_i, gamma_i')
                                - mortality rate in and out of hospitals (d_i, d_i')

         state0 (np.array) : The initial state, a 6 * N array, where N is the number of individuals.

                 T (float) : The end time of the integration.

        t_range (np.array) : A list of times at which the model state is returned. In other words,
                             the output is (6 * N) by len(t_range)

            dt_max (float) : maximum time step?
        '''

        self.model.init(beta = np.exp(params[0]))

        self.model.set_solver(method='RK45', T = T, dt = dt_max)
        ke = self.model.solve(state0, t_range)
        # ke.y is (6 * N) by len(t_range)
        return ke.y

    def backwards_solve(self, params, stateT0, T0, T, dt_max, t_range):
        '''
         params (np.array) : The list of parameters to be inferred, such as (?)
                                - probability of infection / transmission rate (beta_ij)
                                - latent period (1 / sigma_i)
                                - duration of infectiousness in and out of hospitals (gamma_i, gamma_i')
                                - mortality rate in and out of hospitals (d_i, d_i')

         stateT0 (np.array) : The initial state (at T0), a 6 * N array, where N is the number of individuals.

                T0 (Float) : The initial time of integration. 

                 T (float) : The final time of the integration. NOTE: (T<T0)

        t_range (np.array) : A list of times at which the model state is returned. In other words,
                             the output is (6 * N) by len(t_range)

            dt_max (float) : maximum time step?
        '''
        #We integrate backwards equations T0 -> T, so, 0 -> Tback
        Tback=T0-T
        tback_range=np.flip(T0-t_range)

        self.model.init(beta = np.exp(params[0]))

        self.model.set_backwards_solver(method = 'RK45', T = Tback, dt = dt_max)
        ke = self.model.backwards_solve(stateT0, tback_range)

        # ke.y is (6 * N) by len(t_range)
        return ke.y
