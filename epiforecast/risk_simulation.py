import scipy.sparse as sps
import numpy as np
import networkx as nx
from tqdm.autonotebook import tqdm

class NetworkCompartmentalModel(object):
    """
        Model class
    """

    def __init__(self, contact_network = contact_network, ix_reduced = True):
        self.G = self.contact_network = contact_network
        self.N = len(G)

    def set_parameters(self, transmission_rate = 0.06,
                             transition_rates  = transition_rates,
                             ix_reduced        = True):
        """
            Setup the parameters from the transition_rates container into the
            model instance. The default behavior is the reduced model with 5
            equations per node.
        """
        self.beta   = transmission_rate
        self.betap  = 0.75 * self.beta


        self.sigma  = np.array(transition_rates.exposed_to_infected.values())
        self.delta  = np.array(transition_rates.infected_to_hospitalized.values())
        self.theta  = np.array(transition_rates.infected_to_resistant.values())
        self.thetap = np.array(transition_rates.hospitalized_to_resistant.values())
        self.mu     = np.array(transition_rates.infected_to_deceased.values())
        self.mup    = np.array(transition_rates.hospitalized_to_deceased.values())

        self.gamma  = self.theta  + self.mu  + self.delta
        self.gammap = self.thetap + self.mup

        self.L      = nx.to_scipy_sparse_matrix(self.G)

        if ix_reduced:
            iS, iI, iH = [range(jj * self.N, (jj + 1) * self.N) for jj in range(3)]
            self.coeffs = sps.csr_matrix(sps.bmat(
                [
                    [-sps.eye(self.N),       None                                None,                    None,                   None],
                    [sps.diags(-self.sigma), sps.diags(-self.sigma -self.gamma), sps.diags(-self.sigma),  sps.diags(-self.sigma), sps.diags(-self.sigma)],
                    [None,                   sps.diags(self.delta),              sps.diags(-self.gammap), None,                   None],
                    [None,                   sps.diags(self.theta),              sps.diags(self.thetap),  None,                   None],
                    [None,                   sps.diags(self.mu),                 sps.diags(self.mup),     None,                   None]
                ],
                format = 'csr'), shape = [5 * self.N, 5 * self.N])
            self.offset = np.zeros(5 * self.N,)
            self.offset[iI] = self.sigma
            self.y_dot = np.zeros_like(5 * self.N,)
        else:
            print('Warning! Full system not yet implemented')
            pass

    def get_parameters(self, parameter_names = 'All'):
        if parameter_names == 'All':
            params = np.array([self.sigma, self.delta, self.theta, self.thetap, self.mu, self.mup])
        else:
            # Maybe we only want to to the filtering for a subset of params?
            # TODO: Check with Jinlong and Ollie
            pass
        return params

    def update_parameters(self, parameters, parameter_names = 'All'):
        if parameter_names == 'All':
            self.sigma, self.delta, self.theta, self.thetap, self.mu, self.mup = get_parameters
            self.gamma  = self.theta  + self.mu  + self.delta
            self.gammap = self.thetap + self.mup
        else:
            # Maybe we only want to to the filtering for a subset of params?
            # TODO: Check with Jinlong and Ollie
            pass

class MasterEquationModelEnsemble(object):
    def __init__(self, ensemble_size          = 1,
                       contact_network        = contact_network,
                       transmission_rates     = constant_transmission_rate,
                       state_transition_rates = transition_rates):
        self.M = ensemble_size
        self.G = contact_network
        self.ensemble = []

        for mm in tqdm(range(self.G), desc = 'Building ensemble', total = self.M):
            member = NetworkCompartmentalModel(contact_network = G)
            member.set_parameters(transmission_rate = constant_transmission_rate,
                                   transition_rates = transition_rates[mm])
            self.ensemble.append(member)

        self.PM = np.identity(self.M) - 1./self.M * np.ones([self.M,self.M])
        self.L  = nx.to_scipy_sparse_matrix(self.G)

    #  Set methods -------------------------------------------------------------

    def set_state(self, y):
        # Set the master equation state to the same initial state as the kinetic model.
        # master_model.set_state(initial_state)
        self.y0 = y

    def set_contact_network(self):
        """
            For update purposes
        """
        pass

    def set_ensemble(self, new_ensemble):
        """
            For update purposes
        """
        pass

    def set_transition_rates(self, new_transition_rates):
        """
            For update purposes
        """
        pass

    def set_transmission_rates(self, new_transmission_rates):
        """
            For update purposes
        """
        pass

    def populate_states(self, distribution=state_distribution):
        pass

    def populate_transition_rates(self, distribution=transition_rates_distribution):
        pass

    def populate_transmission_rates(self, distribution=transmission_rates_distribution):
        pass

    # Ode solver methods -------------------------------------------------------
    def do_step(self, t, y, member, member_id, **kwargs):
        """
            Inputs:
            y (array): an array of dims (M times N_statuses times N_nodes)
            t (array): times for ode solver

            Returns:
            y_dot (array): lhs of master eqns
        """
        iS, iI, iH = [range(jj * member.N, (jj + 1) * member.N) for jj in range(3)]
        member.beta_closure_ind = sps.kron(np.array([member.beta, member.betap]), self.L).dot(y[iI[0]:(iH[-1]+1)])

        if kwargs.get('closure', 'individual') == 'independent':
            member.yS_holder = self.beta_closure_indp[:, member_id] * y[iS]
        else:
            member.yS_holder = member.beta_closure_ind * y[iS]

        member.y_dot     =   member.coeffs.dot(y) + member.offset
        member.y_dot[iS] = - member.yS_holder
        # member.y_dot[  (member.y_dot > y/self.dt)   & ((member.y_dot < 0) &  (y < 1e-12)) ] = 0.
        # member.y_dot[(member.y_dot < (1-y)/self.dt) & ((member.y_dot > 0) & (y > 1-1e-12))] = 0.

        return member.y_dot

    def eval_closure(self, y, **kwargs):
        iS, iI, iH     = [range(jj * self.N, (jj + 1) * self.N) for jj in range(3)]

        if kwargs.get('closure', 'individual') == 'independent':
            self.numSI = y[:,iS].T.dot(y[:,iI])/(self.M)
            self.denSI = y[:,iS].mean(axis = 0).reshape(-1,1).dot(y[:,iI].mean(axis = 0).reshape(1,-1))

            self.numSH = y[:,iS].T.dot(y[:,iH])/(self.M)
            self.denSH = y[:,iS].mean(axis = 0).reshape(-1,1).dot(y[:,iH].mean(axis = 0).reshape(1,-1))
            self.beta_closure_indp = self.beta  * self.L.multiply((self.numSI/(self.denSI+1e-8))).dot(y[:,iI].T) + \
                                     self.betap * self.L.multiply((self.numSH/(self.denSH+1e-8))).dot(y[:,iH].T)

    def simulate(self, y0, T, n_steps = 100, t0 = 0.0, args = (), **kwargs):
        """
        """
        self.tf = 0.
        self.y0 = np.copy(y0)
        t       = np.linspace(t0, T, nsteps + 1)
        self.dt = np.diff(t).min()
        yt      = np.empty((len(y0.flatten()), len(t)))
        yt[:,0] = np.copy(y0.flatten())

        for jj, time in tqdm(enumerate(t[:-1]), desc = 'Simulate forward', total = n_steps):
            self.eval_closure(self.y0, **kwargs)
            for mm, member in enumerate(self.ensemble):
                self.y0[mm] += self.dt * self.ens_keqns_sparse_closure(t, self.y0[mm], member, mm, **kwargs)
                self.y0[mm] = np.clip(self.y0[mm], 0., 1.)
            self.tf += self.dt
            yt[:,jj + 1] = np.copy(self.y0.flatten())

        return yt.reshape(self.M, -1, len(t))
