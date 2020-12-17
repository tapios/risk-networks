import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix
from multiprocessing import shared_memory
from timeit import default_timer as timer

import ray
from numba import jit 

@jit(nopython=True)
def create_CM_data(nonzeros, rows, cols, data, M, yS, yI, yH, Smean, Imean, Hmean):
    CM_SI_data = np.zeros(nonzeros)
    CM_SH_data = np.zeros(nonzeros)
    for k,i,j,v in zip(range(nonzeros), rows, cols, data):
        #for SI interactions
        CM_SI_data[k] = yS[:,i].dot(yI[:,j])/M  #create bar{<S_i,I_j>}
        CM_SI_data[k] /= Smean[i]*Imean[j]+1e-8 #divide by <S_i><I_j>+eps
        CM_SI_data[k] *= v                      #multiply by the weight matrix wij
        
        #for SH interactions
        CM_SH_data[k] = yS[:,i].dot(yH[:,j])/M
        CM_SH_data[k] /= Smean[i]*Hmean[j]+1e-8
        CM_SH_data[k] *= v
    
    return CM_SI_data, CM_SH_data
 
class MasterEquationModelEnsemble:
    def __init__(
            self,
            population,
            transition_rates,
            transmission_rate,
            hospital_transmission_reduction=0.25,
            ensemble_size=1,
            start_time=0.0,
            parallel_cpu=False,
            num_cpus=1):
        """
        Constructor

        Input:
            population (int): population count
            transition_rates (TransitionRates): same rates for each ensemble
                                                member
                             (list of TransitionRates): list of length
                                                        ensemble_size with
                                                        individual rates for
                                                        each member
            transmission_rate (list): list of length ensemble_size with
                                      individual rate for each member
                              (np.array): (M, 1) array of individual rates
            hospital_transmission_reduction (float): fraction of beta in
                                                     hospitals
            ensemble_size (int): number of ensemble members
            start_time (float): start time of the simulation
            parallel_cpu (bool): whether to run computation in parallel on CPU
            num_cpus (int): number of CPUs available; only used in parallel mode
        """
        assert len(transmission_rate) == ensemble_size

        self.M = ensemble_size
        self.N = population
        self.start_time = start_time

        self.S_slice = slice(       0,   self.N)
        self.I_slice = slice(  self.N, 2*self.N)
        self.H_slice = slice(2*self.N, 3*self.N)
        self.R_slice = slice(3*self.N, 4*self.N)
        self.D_slice = slice(4*self.N, 5*self.N)

        self.hospital_transmission_reduction = hospital_transmission_reduction

        self.parallel_cpu = parallel_cpu
        if parallel_cpu:
            # need to save SharedMemory objects b/c otherwise they're
            # garbage-collected at the end of constructor

            float_nbytes = np.dtype(np.float_).itemsize

            y0_nbytes = self.M * self.N * 5 * float_nbytes
            self.y0_shm = shared_memory.SharedMemory(create=True,
                                                     size=y0_nbytes)
            self.y0 = np.ndarray((self.M, self.N * 5),
                                 dtype=np.float_,
                                 buffer=self.y0_shm.buf)

            closure_nbytes = self.M * self.N * float_nbytes
            self.closure_shm = shared_memory.SharedMemory(create=True,
                                                          size=closure_nbytes)
            self.closure = np.ndarray((self.M, self.N),
                                      dtype=np.float_,
                                      buffer=self.closure_shm.buf)

            coefficients_nbytes = self.M * self.N * 8 * float_nbytes
            self.coefficients_shm = shared_memory.SharedMemory(
                    create=True,
                    size=coefficients_nbytes)
            self.coefficients = np.ndarray((self.M, 8*self.N),
                                           dtype=np.float_,
                                           buffer=self.coefficients_shm.buf)

            members_chunks = np.array_split(np.arange(self.M), num_cpus)
            self.integrators = [
                    RemoteIntegrator.remote(members_chunks[j],
                                            self.M,
                                            self.N,
                                            self.y0_shm.name,
                                            self.coefficients_shm.name,
                                            self.closure_shm.name)
                    for j in range(num_cpus)
            ]
        else:
            self.y0           = np.empty( (self.M, 5*self.N) )
            self.closure      = np.empty( (self.M,   self.N) )
            self.coefficients = np.empty( (self.M, 8*self.N) )

        self.update_transition_rates(transition_rates)

        n_beta_per_member = 1
        self.ensemble_beta_infected = np.empty( (self.M, n_beta_per_member) )
        self.ensemble_beta_hospital = np.empty( (self.M, n_beta_per_member) )
        self.update_transmission_rate(transmission_rate)

        self.walltime_eval_closure = 0.0

    def __extract_coefficients(
            self,
            transition_rates):
        """
        Extract coefficients from TransitionRates object into np.array

        Input:
            transition_rates (TransitionRates): transition rates of a member
        Output:
            coefficients (np.array): (8*N,) array of master equations coeffs
        """
        sigma= transition_rates.get_transition_rate('exposed_to_infected')
        delta= transition_rates.get_transition_rate('infected_to_hospitalized')
        xi   = transition_rates.get_transition_rate('infected_to_resistant')
        xip  = transition_rates.get_transition_rate('hospitalized_to_resistant')
        mu   = transition_rates.get_transition_rate('infected_to_deceased')
        mup  = transition_rates.get_transition_rate('hospitalized_to_deceased')

        gamma  = xi  + mu  + delta
        gammap = xip + mup

        return np.hstack((
                sigma,
                gamma,
                delta,
                xi,
                mu,
                gammap,
                xip,
                mup
        ))

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
        Output:
            None
        """
        self.L = mean_contact_duration

    def update_transmission_rate(
            self,
            transmission_rate):
        """
        Set transmission rates a.k.a. betas

        Input:
            transmission_rate (np.array): (M, 1) array of rates
                              (list): list of rates of length M
        Output:
            None
        """
        if isinstance(transmission_rate, list):
            self.ensemble_beta_infected[:,0] = np.fromiter(transmission_rate,
                                                           dtype=np.float_)
        else:
            self.ensemble_beta_infected[:] = transmission_rate

        self.ensemble_beta_hospital[:] = (
                self.hospital_transmission_reduction *
                self.ensemble_beta_infected
        )

    def update_transition_rates(
            self,
            transition_rates):
        """
        Set transition rates a.k.a. sigma, gamma etc.

        Input:
            transition_rates (TransitionRates): same rates for each ensemble
                                                member
                             (list of TransitionRates): list of length
                                                        ensemble_size with
                                                        individual rates for
                                                        each member
        Output:
            None
        """
        if isinstance(transition_rates, list):
            for j in range(self.M):
                self.coefficients[j] = self.__extract_coefficients(
                        transition_rates[j])
        else:
            # XXX obviously, there's memory overhead here; for simplicity's sake
            coefficients = self.__extract_coefficients(transition_rates)
            for j in range(self.M):
                self.coefficients[j] = coefficients

    def update_ensemble(
            self,
            new_transition_rates,
            new_transmission_rate):
        """
        Set all parameters of the ensemble (transition and transmission rates)

        Input:
            new_transition_rates (TransitionRates): same rates for each ensemble
                                                    member
                                 (list of TransitionRates): list of length
                                                            ensemble_size with
                                                            individual rates for
                                                            each member
            new_transmission_rate (np.array): (M, 1) array of rates
                                  (list): list of rates of length M
        Output:
            None
        """
        self.update_transition_rates(new_transition_rates)
        self.update_transmission_rate(new_transmission_rate)

    def set_states_ensemble(
            self,
            states_ensemble):
        self.y0[:] = states_ensemble[:]

    def compute_rhs(
            self,
            j,
            member_state):
        """
        Compute right-hand side of master equations

        Input:
            j (int): index of the ensemble member
            member_state (np.array): (5*N,) array of states
        Output:
            rhs (np.array): (5*N,) right-hand side of master equations
        """
        S_substate = member_state[self.S_slice]
        I_substate = member_state[self.I_slice]
        H_substate = member_state[self.H_slice]
        E_substate = 1 - member_state.reshape( (5, -1) ).sum(axis=0)

        (sigma,
         gamma,
         delta,
         xi,
         mu,
         gammap,
         xip,
         mup
        ) = self.coefficients[j].reshape( (8, -1) )

        rhs = np.empty(5 * self.N)

        rhs[self.S_slice] = -self.closure[j] * S_substate
        rhs[self.I_slice] = sigma * E_substate - gamma  * I_substate
        rhs[self.H_slice] = delta * I_substate - gammap * H_substate
        rhs[self.R_slice] = xi    * I_substate + xip    * H_substate
        rhs[self.D_slice] = mu    * I_substate + mup    * H_substate

        return rhs

    def eval_closure(
            self,
            closure_name):
        """
        Evaluate closure from full ensemble state 'self.y0'

        Input:
            closure_name (str): which closure to evaluate; only 'independent'
                                and 'full' are supported at this time
        Output:
            None
        """
        if closure_name == 'independent':
            iS, iI, iH = self.S_slice, self.I_slice, self.H_slice
            y = self.y0

            S_ensemble_mean = y[:,iS].mean(axis=0)
            I_ensemble_mean = y[:,iI].mean(axis=0)
            H_ensemble_mean = y[:,iH].mean(axis=0)
            yS = y[:,iS]
            yI = y[:,iI]
            yH = y[:,iH]

            cooL = self.L.tocoo()
            nonzeros = len(cooL.row)
            cooL_rows = cooL.row
            cooL_cols = cooL.col
            cooL_data = cooL.data

            CM_SI_data,CM_SH_data = create_CM_data(
                nonzeros,
                cooL_rows,
                cooL_cols,
                cooL_data, 
                self.M,
                yS,
                yI,
                yH,
                S_ensemble_mean,
                I_ensemble_mean,
                H_ensemble_mean)

            CM_S = coo_matrix((np.array(CM_SI_data),(cooL.row,cooL.col)),shape=(self.N,self.N)).tocsr()
            CM_S @= y[:,iI].T
            self.closure[:] =  CM_S.T * self.ensemble_beta_infected

            CM_S = coo_matrix((np.array(CM_SH_data),(cooL.row,cooL.col)),shape=(self.N,self.N)).tocsr()
            CM_S @= y[:,iH].T
            self.closure[:] +=  CM_S.T * self.ensemble_beta_hospital

        elif closure_name == 'full':
            # XXX this only works for betas of shape (M, 1); untested for others
            ensemble_I_substate = self.y0[:, self.I_slice] # (M, N)
            ensemble_H_substate = self.y0[:, self.H_slice] # (M, N)

            closure_I = ensemble_I_substate @ self.L
            closure_H = ensemble_H_substate @ self.L
            self.closure[:] = (  closure_I * self.ensemble_beta_infected,
                               + closure_H * self.ensemble_beta_hospital)
        else:
            raise ValueError(
                    self.__class__.__name__
                    + ": this value of 'closure_name' is not supported: "
                    + closure_name)

    def simulate(
            self,
            time_window,
            min_steps=1,
            closure_name='independent',
            closure_flag=True):
        """
        Simulate master equations for the whole ensemble forward in time

        Input:
            time_window (float): duration of simulation
            min_steps (int): minimum number of timesteps
            closure_name (str): which closure to use; only 'independent' and
                                'full' are supported at this time
            closure_flag (bool): whether to evaluate closure during this
                                 simulation call
        Output:
            y0 (np.array): (M, 5*N) array of states at the end of time_window
        """
        stop_time = self.start_time + time_window
        maxdt = abs(time_window) / min_steps

        if closure_flag:
            timer_eval_closure = timer()
            self.eval_closure(closure_name)
            self.walltime_eval_closure += timer() - timer_eval_closure

        if self.parallel_cpu:
            futures = []
            args = (self.start_time, stop_time, maxdt)

            for integrator in self.integrators:
                futures.append(integrator.integrate.remote(*args))

            for future in futures:
                ray.get(future)
        else:
            for j in range(self.M):
                ode_result = solve_ivp(
                        fun = lambda t, y: (
                            self.compute_rhs(j, y)
                            ),
                        t_span = [self.start_time, stop_time],
                        y0 = self.y0[j],
                        t_eval = [stop_time],
                        method = 'RK45',
                        max_step = maxdt)

                self.y0[j] = np.clip(np.squeeze(ode_result.y), 0, 1)

        self.start_time += time_window
        return self.y0

    def simulate_backwards(
            self,
            time_window,
            min_steps=1,
            closure_name='independent',
            closure_flag=True):

        """
        Simulate master equations for the whole ensemble backward in time

        Input:
            time_window (float): duration of simulation
            min_steps (int): minimum number of timesteps
            closure_name (str): which closure to use; only 'independent' and
                                'full' are supported at this time
            closure_flag (bool): whether to evaluate closure during this
                                 simulation call
        Output:
            y0 (np.array): (M, 5*N) array of states at the end of time_window
        """
        positive_time_window = abs(time_window)
        
        return self.simulate(-positive_time_window,
                             min_steps,
                             closure_name,
                             closure_flag)

    def reset_walltimes(self):
        """
        Reset walltimes to zero
        """
        self.walltime_eval_closure = 0.0

    def get_walltime_eval_closure(self):
        """
        Get walltime of the 'eval_closure' calls
        """
        return self.walltime_eval_closure

    def wrap_up(self):
        """
        Take care of the shared memory objects

        This should be called only once at the end of object's lifetime;
        unfortunately, destructor doesn't work for this (some memory is already
        being freed, so it throws an exception).

        Output:
            None
        """
        if self.parallel_cpu:
            self.y0_shm.close()
            self.closure_shm.close()
            self.coefficients_shm.close()

            self.y0_shm.unlink()
            self.closure_shm.unlink()
            self.coefficients_shm.unlink()


@ray.remote
class RemoteIntegrator:
    def __init__(
            self,
            members_to_compute,
            M,
            N,
            ensemble_state_shared_memory_name,
            coefficients_shared_memory_name,
            closure_shared_memory_name):
        """
        Constructor

        Input:
            members_to_compute (np.array): indices of members that are assigned
                                           to this integrator
            M (int): total number of ensemble members
            N (int): total number of nodes (population)
            ensemble_state_shared_memory_name (str): shared memory name (state)
            coefficients_shared_memory_name (str): shared memory name (coeffs)
            closure_shared_memory_name (str): shared memory name (closure)
        """
        self.M = M
        self.N = N
        self.shared_memory_names = {
                'ensemble_state': ensemble_state_shared_memory_name,
                'coefficients'  : coefficients_shared_memory_name,
                'closure'       : closure_shared_memory_name
        }

        self.S_slice = slice(  0,   N)
        self.I_slice = slice(  N, 2*N)
        self.H_slice = slice(2*N, 3*N)
        self.R_slice = slice(3*N, 4*N)
        self.D_slice = slice(4*N, 5*N)

        self.members_to_compute = members_to_compute

    def integrate(
            self,
            start_time,
            stop_time,
            maxdt):
        """
        Integrate members assigned to the current integrator

        All three arguments are arguments to scipy.integrate.solve_ivp

        Input:
            start_time (float): start time of the integration interval
            stop_time (float): stop time of the integration interval
            maxdt (float): maximum timestep
        Output:
            None
        """
        y0_shm = shared_memory.SharedMemory(
                name=self.shared_memory_names['ensemble_state'])
        coefficients_shm = shared_memory.SharedMemory(
                name=self.shared_memory_names['coefficients'])
        closure_shm = shared_memory.SharedMemory(
                name=self.shared_memory_names['closure'])

        ensemble_state = np.ndarray((self.M, 5*self.N),
                                    dtype=np.float_,
                                    buffer=y0_shm.buf)
        coefficients   = np.ndarray((self.M, 8*self.N),
                                    dtype=np.float_,
                                    buffer=coefficients_shm.buf)
        closure        = np.ndarray((self.M, self.N),
                                    dtype=np.float_,
                                    buffer=closure_shm.buf)

        t_span = np.array([start_time, stop_time])
        t_eval = np.array([stop_time])

        for j in self.members_to_compute:
            member_state        = ensemble_state[j]
            member_coefficients = coefficients[j]
            member_closure      = closure[j]

            compute_rhs_member = lambda t, y: (
                    self.compute_rhs(y, member_coefficients, member_closure)
            )

            ode_result = solve_ivp(fun=compute_rhs_member,
                                   t_span=t_span,
                                   y0=member_state,
                                   t_eval=t_eval,
                                   method='RK45',
                                   max_step=maxdt)
            ensemble_state[j] = np.clip(np.squeeze(ode_result.y), 0.0, 1.0)

        y0_shm.close()
        coefficients_shm.close()
        closure_shm.close()

        return

    def compute_rhs(
            self,
            member_state,
            member_coefficients,
            member_closure):
        """
        Compute right-hand side of master equations

        Input:
            member_state (np.array): (5*N,) array of states
            member_coefficients (np.array): (8*N,) array of coefficients of the
                                            linear part of the RHS
            member_closure (np.array): (N,) array of coefficients for S_i's
        Output:
            rhs (np.array): (5*N,) right-hand side of master equations
        """
        S_substate = member_state[self.S_slice]
        I_substate = member_state[self.I_slice]
        H_substate = member_state[self.H_slice]
        E_substate = 1 - member_state.reshape( (5, -1) ).sum(axis=0)

        (sigma,
         gamma,
         delta,
         xi,
         mu,
         gammap,
         xip,
         mup
        ) = member_coefficients.reshape( (8, -1) )

        rhs = np.empty(5*self.N)

        rhs[self.S_slice] = -member_closure * S_substate
        rhs[self.I_slice] = sigma * E_substate - gamma  * I_substate
        rhs[self.H_slice] = delta * I_substate - gammap * H_substate
        rhs[self.R_slice] = xi    * I_substate + xip    * H_substate
        rhs[self.D_slice] = mu    * I_substate + mup    * H_substate

        return rhs


