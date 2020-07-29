import scipy.sparse as sps
from scipy.integrate  import solve_ivp, odeint
from epiforecast.risk_simulator_integrator import MasterEquationModelIntegrator
import numpy as np
import networkx as nx
import copy
import time
from pathos.pools import ProcessPool 

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

        self.pool = ProcessPool()

        for mm in range(self.M):
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
            closure = 'independent',
            parallel_flag = True):
        """
        Args:
        -------
        time_window : duration of simulation
          min_steps : minimum number of timesteps
            closure : by default consider that closure = 'independent'
        """
        stop_time = self.start_time + time_window
        maxdt = abs(time_window) / min_steps

        self.eval_closure(self.y0, closure = closure)

        if parallel_flag == True:
            results = []
            for mm, member in enumerate(self.ensemble):
                y0 = np.copy(self.y0[mm])
                start_time = self.start_time
                integrator_obj = MasterEquationModelIntegrator()
                if closure == 'independent':
                    integrator_obj.set_values(np.copy(self.L), 
                                              member, closure, 
                                              np.copy(self.CM_SI[:,mm]), 
                                              np.copy(self.CM_SH[:,mm]))
                else:
                    integrator_obj.set_values(self.L, member)
                results.append(self.pool.apipe(integrator_obj.integrator,
                                                closure,
                                                start_time, 
                                                stop_time,
                                                y0,
                                                maxdt))
            mm = 0
            for single_result in results:
                result = single_result.get()
                self.y0[mm] = np.clip(np.squeeze(result.y),0,1)
                mm += 1

        else:
            for mm, member in enumerate(self.ensemble):
                y0 = self.y0[mm]
                start_time = self.start_time
                integrator_obj = MasterEquationModelIntegrator()
                if closure == 'independent':
                    integrator_obj.set_values(self.L, 
                                              member, closure,
                                              self.CM_SI[:,mm], self.CM_SH[:,mm])
                else:
                    integrator_obj.set_values(self.L, member)
                result = integrator_obj.integrator(closure,
                                                    start_time, stop_time,
                                                    y0,
                                                    maxdt)
   
                self.y0[mm] = np.clip(np.squeeze(result.y),0,1)

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
        
        return self.simulate(-positive_time_window,
                             min_steps,
                             closure)

