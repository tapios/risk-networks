import scipy.sparse as sps
import numpy as np
import networkx as nx
from tqdm.autonotebook import tqdm

class NetworkCompartmentalModel:
    """
    ODE representation of the SEIHRD compartmental model.
    """

    def __init__(
            self,
            N,
            user_connectivity,
            hospital_transmission_reduction = 0.25):

        self.hospital_transmission_reduction = hospital_transmission_reduction
        self.user_connectivity = user_connectivity
        self.N = N

    def set_parameters(
            self,
            transition_rates  = None,
            transmission_rate = None,
            exogenous_transmission_rate = None):
        """
        Setup the parameters from the transition_rates, transmission rates and exogenous transmission_rate
        container into the model instance.
        """

        if transmission_rate is not None:
            self.beta   = transmission_rate 
            self.betap  = self.hospital_transmission_reduction * self.beta

        if exogenous_transmission_rate is not None:
            self.eta = exogenous_transmission_rate * self.user_connectivity
           

        if transition_rates is not None:
            self.sigma  = np.array(list(transition_rates.exposed_to_infected.values()))
            self.delta  = np.array(list(transition_rates.infected_to_hospitalized.values()))
            self.theta  = np.array(list(transition_rates.infected_to_resistant.values()))
            self.thetap = np.array(list(transition_rates.hospitalized_to_resistant.values()))
            self.mu     = np.array(list(transition_rates.infected_to_deceased.values()))
            self.mup    = np.array(list(transition_rates.hospitalized_to_deceased.values()))

            self.gamma  = self.theta  + self.mu  + self.delta
            self.gammap = self.thetap + self.mup

            # Reduced system with 5 equations ------------------------------
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

    def update_transition_rates(
            self,
            new_transition_rates):
        """
        Args:
        -------
        transition_rates dictionary.
        """
        self.set_parameters(transition_rates  = new_transition_rates,
                            transmission_rate = None,
                  exogenous_transmission_rate = None)

    def update_transmission_rate(
            self,
            new_transmission_rate):
        """
        Args:
        -------
        new_transmission_rate np.array.
        """
        self.set_parameters(transition_rates  = None,
                            transmission_rate = new_transmission_rate,
                  exogenous_transmission_rate = None )
        
    def update_exogenous_transmission_rate(
            self,
            new_exogenous_transmission_rate):
        """
        Args:
        -------
        new_exogenous_transmission_rate : np.array.
        """
        self.set_parameters(transition_rates  = None,
                            transmission_rate = None,
                  exogenous_transmission_rate = new_exogenous_transmission_rate)

        
class MasterEquationModelEnsemble:
    def __init__(
            self,
            contact_network,
            transition_rates,
            transmission_rate,
            ensemble_size = 1,
            exogenous_transmission_rate = None,
            hospital_transmission_reduction = 0.25,
            start_time = 0.0):
        """
        Args:
        -------
            ensemble_size : `int`
          contact_network : `networkx.graph.Graph` or Weighted adjacency matrix `scipy.sparse.csr_matrix`
         transition_rates : `list` or single instance of `TransitionRate` container
        transmission_rate : `np.array` length ensemble size
        exogenous_transmission_rate    : `np.array` length ensemble size  (defined for when user network smaller than contact network)
        """

       
        
        self.G = self.contact_network = contact_network
        self.M = self.ensemble_size = ensemble_size
        self.N = len(self.G)
        self.start_time = start_time

        if exogenous_transmission_rate is None:
            self.user_connectivity = np.ones(self.N)
        else:
            user_connectivity = nx.get_node_attributes(contact_network, name = 'user_connectivity')
            self.user_connectivity = np.array([user_connectivity[node] for node in self.G.nodes])
        
        if exogenous_transmission_rate is None:
            if isinstance(transition_rates, list):
                exogenous_transmission_rate = np.zeros(ensemble_size)
            else:
                exogenous_transmission_rate = 0.0
           
        
        self.ensemble = []

        for mm in tqdm(range(self.M), desc = 'Building ensemble', total = self.M):
            member = NetworkCompartmentalModel(N = self.N,
                                               user_connectivity = self.user_connectivity,
                                               hospital_transmission_reduction = hospital_transmission_reduction)

            if isinstance(transition_rates, list):
                member.set_parameters(transition_rates  = transition_rates[mm],
                                      transmission_rate = transmission_rate[mm],
                            exogenous_transmission_rate = exogenous_transmission_rate[mm])
            else:
                member.set_parameters(transition_rates  = transition_rates,
                                      transmission_rate = transmission_rate,
                            exogenous_transmission_rate = exogenous_transmission_rate)
            member.id = mm
            self.ensemble.append(member)

        self.PM = np.identity(self.M) - 1./self.M * np.ones([self.M,self.M])

    #  Set methods -------------------------------------------------------------
    def set_start_time(
            self,
            start_time):
        self.start_time = start_time

    def set_mean_contact_duration(
            self,
            new_mean_contact_duration=None):
        """
        For update purposes
        """
        if new_mean_contact_duration is not None:
            self.weight = {tuple(edge): new_mean_contact_duration[i]
                           for i, edge in enumerate(nx.edges(self.contact_network))}
            nx.set_edge_attributes(self.contact_network, values=self.weight, name='exposed_by_infected')

        self.L = nx.to_scipy_sparse_matrix(self.contact_network, weight = 'exposed_by_infected')

    def set_contact_network_and_contact_duration(
            self,
            new_contact_network):
        self.contact_network = new_contact_network
        # Automatically reset the edge weights
        self.set_mean_contact_duration()

    def update_transmission_rate(
            self,
            new_transmission_rate):
        """
        new_transmission_rate : `np.array` of length `ensemble_size`
        """
        for mm, member in enumerate(self.ensemble):
            member.update_transmission_rate(new_transmission_rate[mm])

    def update_exogenous_transmission_rate(
            self,
            new_exogenous_transmission_rate):
        """
        new_transmission_rate : `np.array` of length `ensemble_size`
        new_exogenous_transmission_rate : `np.array` of length `ensemble_size`
        
        """
        for mm, member in enumerate(self.ensemble):
            member.update_exogenous_transmission_rate(new_exogenous_transmission_rate[mm])

    def update_transition_rates(self,
                                new_transition_rates):
        """
        new_transition_rates : `list` of `TransitionRate`s
        """
        for mm, member in enumerate(self.ensemble):
            member.update_transition_rates(new_transition_rates[mm])

    def update_ensemble(
            self,
            new_transition_rates,
            new_transmission_rate,
            new_exogenous_transmission_rate):
        """
        update all parameters of ensemeble
        """
        self.update_transition_rates(new_transition_rates)
        self.update_transmission_rate(new_transmission_rate)
        self.update_exogenous_transmission_rate(new_exogenous_transmission_rate)

    def set_states_ensemble(
            self,
            states_ensemble):
        self.y0 = np.copy(states_ensemble)

    # ODE solver methods -------------------------------------------------------
    def do_step(
            self,
            t,
            y,
            member,
            closure='independent'):
        """
        Args:
        --------
        y (array): an array of dims (M times N_statuses times N_nodes)
        t (array): times for ode solver

        Returns:
        --------
        y_dot (array): lhs of master eqns
        """
        iS, iI, iH = [range(jj * member.N, (jj + 1) * member.N) for jj in range(3)]

        if closure == 'independent':
            member.beta_closure = member.beta  * self.CM_SI[:, member.id] + \
                                  member.betap * self.CM_SH[:, member.id]
        else:
            member.beta_closure = (sps.kron(np.array([member.beta, member.betap]), self.L).T.dot(y[iI[0]:(iH[-1]+1)])).T
            member.yS_holder = member.beta_closure  * y[iS]

        #Add in the exogenous_transmission infections here?
        member.yS_holder = (member.beta_closure + member.eta)*y[iS]
            
        member.y_dot     =   member.coeffs.dot(y) + member.offset
        member.y_dot[iS] = - member.yS_holder
        # member.y_dot[  (member.y_dot > y/self.dt)   & ((member.y_dot < 0) &  (y < 1e-12)) ] = 0.
        # member.y_dot[(member.y_dot < (1-y)/self.dt) & ((member.y_dot > 0) & (y > 1-1e-12))] = 0.

        return member.y_dot

    def eval_closure(self,
                     y,
                     closure='independent'):

        iS, iI, iH = [range(jj * self.N, (jj + 1) * self.N) for jj in range(3)]

        if closure == 'independent':
            self.numSI = y[:,iS].T.dot(y[:,iI])/(self.M)
            self.denSI = y[:,iS].mean(axis = 0).reshape(-1,1).dot(y[:,iI].mean(axis = 0).reshape(1,-1))

            self.numSH = y[:,iS].T.dot(y[:,iH])/(self.M)
            self.denSH = y[:,iS].mean(axis = 0).reshape(-1,1).dot(y[:,iH].mean(axis = 0).reshape(1,-1))

            self.CM_SI = self.L.multiply(self.numSI/(self.denSI+1e-8)).dot(y[:,iI].T)
            self.CM_SH = self.L.multiply(self.numSH/(self.denSH+1e-8)).dot(y[:,iH].T)

    def simulate(
            self,
            time_window,
            n_steps=50,
            closure='independent',
            **kwargs):
        """
        Args:
        -------
        time_window : duration of simulation
            n_steps : number of Euler steps
            closure : by default consider that closure = 'independent'
        """
        self.stop_time = self.start_time + time_window
        t       = np.linspace(self.start_time, self.stop_time, n_steps + 1)
        self.dt = np.diff(t).min()

        yt      = np.empty([self.y0.size, t.size])
        yt[:,0] = self.y0.flatten()

        for jj, time in tqdm(enumerate(t[:-1]),
                             desc = '[ Master equations ] Time window [%2.3f, %2.3f]'%(self.start_time, self.stop_time),
                             total = t.size - 1):
            self.eval_closure(self.y0, closure = closure)
            for mm, member in enumerate(self.ensemble):
                self.y0[mm] += self.dt * self.do_step(t, self.y0[mm], member, closure = closure)
                self.y0[mm]  = np.clip(self.y0[mm], 0., 1.)
            yt[:,jj + 1] = np.copy(self.y0.flatten())

        self.simulation_time = t
        self.states_trace    = yt.reshape(self.M, -1, t.size)
        self.start_time   += time_window

        return self.y0

    def simulate_backwards(
            self,
            time_window,
            n_steps = 100,
            closure = 'independent',
            **kwargs):
        """    
        Args:
        -------
        time_window : duration of simulation
            n_steps : number of Euler steps
            closure : by default consider that closure = 'independent'
        """
        self.stop_time = self.start_time - time_window
        t       = np.linspace(self.start_time, self.stop_time, n_steps + 1)
        self.dt = np.diff(t).min()
        
        yt      = np.empty([self.y0.size, t.size])
        yt[:,0] = self.y0.flatten()
        
        for jj, time in tqdm(enumerate(t[:-1]),
                             desc = '[ Master equations ] Time window [%2.3f, %2.3f]'%(self.stop_time, self.start_time),
                             total = t.size - 1):
            self.eval_closure(self.y0, closure = closure)
            for mm, member in enumerate(self.ensemble):
                self.y0[mm] += self.dt * self.do_step(t, self.y0[mm], member, closure = closure)
                self.y0[mm]  = np.clip(self.y0[mm], 0., 1.)
            yt[:,jj + 1] = np.copy(self.y0.flatten())

        self.simulation_time = t
        self.states_trace    = yt.reshape(self.M, -1, t.size)
        self.start_time   -= time_window

        return self.y0
