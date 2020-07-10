import scipy.sparse as sps
from scipy.integrate  import solve_ivp
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
            hospital_transmission_reduction=0.25):

        self.hospital_transmission_reduction = hospital_transmission_reduction
        self.N = N

    def set_parameters(
            self,
            transition_rates=None,
            transmission_rate=None):
        """
        Setup the parameters from the transition_rates container into the model
        instance.
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

    def update_transition_rates(self, new_transition_rates):
        """
        Args:
        -------
        transition_rates dictionary.
        """
        self.set_parameters(transition_rates = new_transition_rates,
                           transmission_rate = None)

    def update_transmission_rate(
            self,
            new_transmission_rate):
        """
        Args:
        -------
        transmission_rate np.array.
        """
        self.set_parameters(transition_rates = None,
                           transmission_rate = new_transmission_rate)

class MasterEquationModelEnsemble:
    def __init__(
            self,
            population,
            transition_rates,
            transmission_rate,
            hospital_transmission_reduction = 0.25,
            ensemble_size = 1,
            start_time = 0.0):
        """
        Args:
        -------
            ensemble_size : `int`
         transition_rates : `list` or single instance of `TransitionRate` container
        transmission_rate : `list`
        """

        self.M = ensemble_size
        self.N = population
        self.start_time = start_time

        self.ensemble = []

        for mm in tqdm(range(self.M), desc = 'Building ensemble', total = self.M):
            member = NetworkCompartmentalModel(N = self.N,
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
    def set_start_time(
            self,
            start_time):
        self.start_time = start_time

    def set_mean_contact_duration(
            self,
            mean_contact_duration):
        """
        Set mean contact duration a.k.a. L matrix

        Input:
            mean_contact_duration (scipy.sparse.csr.csr_matrix):
                adjacency matrix
        """
        self.L = mean_contact_duration

    def update_transmission_rate(
            self,
            new_transmission_rate):
        """
        new_transmission_rate : `np.array` of length `M`
        """
        for mm, member in enumerate(self.ensemble):
            member.update_transmission_rate(new_transmission_rate[mm])

    def update_transition_rates(
            self,
            new_transition_rates):
        """
        new_transition_rates : `list` of `TransitionRate`s
        """
        for mm, member in enumerate(self.ensemble):
            member.update_transition_rates(new_transition_rates[mm])

    def update_ensemble(
            self,
            new_transition_rates,
            new_transmission_rate):
        """
        update all parameters of ensemeble
        """
        self.update_transition_rates(new_transition_rates)
        self.update_transmission_rate(new_transmission_rate)

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
            member.yS_holder = member.beta_closure * y[iS]
        else:
            member.beta_closure = (sps.kron(np.array([member.beta, member.betap]), self.L).T.dot(y[iI[0]:(iH[-1]+1)])).T
            member.yS_holder = member.beta_closure  * y[iS]

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
            min_steps = 1,
            closure = 'independent'):
        """
        Args:
        -------
        time_window : duration of simulation
          min_steps : minimum number of timesteps
            closure : by default consider that closure = 'independent'
        """
        
        self.stop_time = self.start_time + time_window
        self.maxdt = abs(time_window) / min_steps

        self.eval_closure(self.y0, closure = closure)

        for mm, member in enumerate(self.ensemble):

            result = solve_ivp(
		fun = lambda t, y: self.do_step(t, y, member, closure = closure),
		t_span = [self.start_time,self.stop_time],
                y0 = self.y0[mm],
		t_eval = [self.stop_time],
                method = 'RK45',
                max_step = self.maxdt)
           
            self.y0[mm] = np.squeeze(result.y)
    
        self.start_time += time_window
        return self.y0

    def simulate_backwards(
            self,
            time_window,
            min_steps = 1,
            closure = 'independent'):
        """
        We run simulate with a negative time_window
        Args:
        -------
        time_window : duration of simulation
          min_steps : minimum_number of time steps (>=1)
            closure : by default consider that closure = 'independent'
        """
        positive_time_window = abs(time_window)
        y0 =  self.simulate(-positive_time_window,
                            min_steps,
                            closure)
        return y0
       
      
        
