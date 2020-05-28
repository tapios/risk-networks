import scipy.sparse as sps
import numpy as np
import networkx as nx
from tqdm.autonotebook import tqdm

class NetworkCompartmentalModel(object):
    """
        Model class
    """

    def __init__(self, contact_network, ix_reduced = True, weight = None,
                hospital_transmission_reduction = 0.25):
        self.hospital_transmission_reduction = hospital_transmission_reduction
        if type(contact_network) == nx.classes.graph.Graph:
            self.G      = self.contact_network = contact_network
            self.N      = len(self.G)
            self.weight = weight
            self.L      = nx.to_scipy_sparse_matrix(self.G, weight = self.weight)
        else:
            self.L      = sps.csr_matrix(contact_network)
            self.N      = self.L.shape[0]

    def set_parameters(self, transition_rates  = None,
                             transmission_rate = None,
                             ix_reduced        = True):
        """
            Setup the parameters from the transition_rates container into the
            model instance. The default behavior is the reduced model with 5
            equations per node.
        """
        if transmission_rate is not None:
            self.beta   = transmission_rate
            self.betap  = self.hospital_transmission_reduction * self.beta

        if transition_rates is not None:
            self.sigma  = np.array(list(transition_rates.exposed_to_infected.values()))
            self.delta  = np.array(list(transition_rates.infected_to_hospitalized.values()))
            self.theta  = np.array(list(transition_rates.infected_to_resistant.values()))
            self.thetap = np.array(list(transition_rates.hospitalized_to_resistant.values()))
            self.mu     = np.array(list(transition_rates.infected_to_deceased.values()))
            self.mup    = np.array(list(transition_rates.hospitalized_to_deceased.values()))

            self.gamma  = self.theta  + self.mu  + self.delta
            self.gammap = self.thetap + self.mup

            if ix_reduced:
                iS, iI, iH = [range(jj * self.N, (jj + 1) * self.N) for jj in range(3)]
                self.coeffs = sps.csr_matrix(sps.bmat(
                    [
                        [-sps.eye(self.N),       None,                               None,                    None,                   None],
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

    def update_transition_rates(self, new_transition_rates):
        """
        Inputs:
        -------
        transition_rates object
        """
        self.set_parameters(transition_rates = new_transition_rates,
                           transmission_rate = None)

    def update_transmission_rate(self, new_transmission_rate):
        """
        Inputs:
        -------
        transmission_rate float
        """
        self.set_parameters(transition_rates = None,
                           transmission_rate = new_transmission_rate)

    def update_contact_network(self, contact_network):
        if type(contact_network) == nx.classes.graph.Graph:
            self.G = self.contact_network = contact_network
            self.L = nx.to_scipy_sparse_matrix(self.G, weight = self.weight)
        else:
            self.L = sps.csr_matrix(contact_network)

class MasterEquationModelEnsemble(object):
    def __init__(self,
                contact_network,
                transition_rates,
                transmission_rate,
                ensemble_size = 1,
                hospital_transmission_reduction = 0.25,
                weight = None):
        """
        Inputs:
        -------
        ensemble_size (int,)
        contact_network: Graph object (networkx.graph) or Weighted adjacency matrix (sparse matrix)
        transition_rates (list of) or (single instance of) TransitionRate container
        transmission_rate (float)
        """
        self.M = self.ensemble_size   = ensemble_size
        if type(contact_network) == nx.classes.graph.Graph:
            self.G = self.contact_network = contact_network
            self.weight = weight
            self.N = len(self.G)
            self.L  = nx.to_scipy_sparse_matrix(self.G, weight = weight)
        else:
            self.L = sps.csr_matrix(contact_network)
            self.N = self.L.shape[0]

        self.ensemble = []

        for mm in tqdm(range(self.M), desc = 'Building ensemble', total = self.M):
            member = NetworkCompartmentalModel(contact_network = contact_network, weight = weight,
                                hospital_transmission_reduction = hospital_transmission_reduction)
            if isinstance(transition_rates, list):
                member.set_parameters(
                        transition_rates  = transition_rates[mm],
                        transmission_rate = transmission_rate[mm])
            else:
                member.set_parameters(
                        transition_rates  = transition_rates,
                        transmission_rate = transmission_rate)
            member.id = mm
            self.ensemble.append(member)

        self.PM = np.identity(self.M) - 1./self.M * np.ones([self.M,self.M])

    #  Set methods -------------------------------------------------------------
    def update_contact_network(self, new_contact_network):
        """
        For update purposes
        """
        if new_contact_network is not None:
            if type(contact_network) == nx.classes.graph.Graph:
                self.G = self.contact_network = new_contact_network
                self.weight = weight
                self.N = len(self.G)
                self.L  = nx.to_scipy_sparse_matrix(self.G, weight = weight)
            else:
                self.L = sps.csr_matrix(new_contact_network)
                self.N = self.L.shape[0]

            [member.update_contact_network(new_contact_network) for member in self.ensemble]

    def update_transmission_rate(self, new_transmission_rate):
        """
        new_transmission_rate (array) of size M
        """
        for mm, member in enumerate(self.ensemble):
            member.update_transmission_rate(new_transmission_rate[mm])

    def update_transition_rates(self, new_transition_rates):
        """
        list of (or single) transition_rates object
        """
        for mm, member in enumerate(self.ensemble):
            member.update_transition_rates(new_transition_rates[mm])

    def update_ensemble(self,
                        new_contact_network,
                        new_transition_rates,
                        new_transmission_rate):
        """
        update all parameters of ensemeble
        """
        self.update_contact_network(new_contact_network)
        self.update_transition_rates(new_transition_rates)
        self.update_transmission_rate(new_transmission_rate)

    # ODE solver methods -------------------------------------------------------
    def do_step(self, t, y, member, closure = 'independent'):
        """
            Inputs:
            y (array): an array of dims (M times N_statuses times N_nodes)
            t (array): times for ode solver

            Returns:
            y_dot (array): lhs of master eqns
        """
        iS, iI, iH = [range(jj * member.N, (jj + 1) * member.N) for jj in range(3)]
        member.beta_closure_ind = sps.kron(np.array([member.beta, member.betap]), self.L).dot(y[iI[0]:(iH[-1]+1)])

        if closure == 'independent':
            member.beta_closure_indp = member.beta  * self.CM_SI[:, member.id] + \
                                       member.betap * self.CM_SH[:, member.id]
            member.yS_holder = member.beta_closure_indp * y[iS]
        else:
            member.yS_holder = member.beta_closure_ind  * y[iS]

        member.y_dot     =   member.coeffs.dot(y) + member.offset
        member.y_dot[iS] = - member.yS_holder
        # member.y_dot[  (member.y_dot > y/self.dt)   & ((member.y_dot < 0) &  (y < 1e-12)) ] = 0.
        # member.y_dot[(member.y_dot < (1-y)/self.dt) & ((member.y_dot > 0) & (y > 1-1e-12))] = 0.

        return member.y_dot

    def eval_closure(self, y, closure = 'independent'):
        iS, iI, iH     = [range(jj * self.N, (jj + 1) * self.N) for jj in range(3)]

        if closure == 'independent':
            self.numSI = y[:,iS].T.dot(y[:,iI])/(self.M)
            self.denSI = y[:,iS].mean(axis = 0).reshape(-1,1).dot(y[:,iI].mean(axis = 0).reshape(1,-1))

            self.numSH = y[:,iS].T.dot(y[:,iH])/(self.M)
            self.denSH = y[:,iS].mean(axis = 0).reshape(-1,1).dot(y[:,iH].mean(axis = 0).reshape(1,-1))

            self.CM_SI = self.L.multiply(self.numSI/(self.denSI+1e-8)).dot(y[:,iI].T)
            self.CM_SH = self.L.multiply(self.numSH/(self.denSH+1e-8)).dot(y[:,iH].T)

    def simulate(self, y0, T, n_steps = 100, t0 = 0.0, closure = 'independent', **kwargs):
        """
        Inputs:
        -------
        y0 (nd array): initial state for simulation of size (M, 5 times N)
        T (float)    : final time of simulation
        n_steps (int): number of Euler steps
        t0 (float)   : initial time of simulation
        closure      : by default consider that closure = 'independent'
        """
        self.tf = 0.
        self.y0 = np.copy(y0)
        t       = np.linspace(t0, T, n_steps + 1)
        self.dt = np.diff(t).min()
        yt      = np.empty((len(y0.flatten()), len(t)))
        yt[:,0] = np.copy(y0.flatten())

        for jj, time in tqdm(enumerate(t[:-1]), desc = 'Simulate forward', total = n_steps):
            self.eval_closure(self.y0, closure = closure)
            for mm, member in enumerate(self.ensemble):
                self.y0[mm] += self.dt * self.do_step(t, self.y0[mm], member, closure = closure)
                self.y0[mm]  = np.clip(self.y0[mm], 0., 1.)
            self.tf += self.dt
            yt[:,jj + 1] = np.copy(self.y0.flatten())

        return {'times' : t,
                'states': yt.reshape(self.M, -1, len(t))}