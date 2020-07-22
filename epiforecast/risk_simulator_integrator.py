import scipy.sparse as sps
from scipy.integrate  import solve_ivp, odeint
import numpy as np
import networkx as nx
import mkl

class MasterEquationModelIntegrator:

    def set_values(self, L, member, closure = 'None', CM_SI=None, CM_SH=None):
        self.L = L
        self.member = member
        self.closure = closure
        
        if closure == 'independent':
            self.CM_SI = CM_SI
            self.CM_SH = CM_SH

    def compute_rhs(
            self,
            t,
            y):
        """
        Args:
        --------
        y (array): an array of dims (M times N_statuses times N_nodes)
        t (array): times for ode solver

        Returns:
        --------
        y_dot (array): rhs of master eqns
        """
        iS, iI, iH = [range(jj * self.member.N, (jj + 1) * self.member.N) for jj in range(3)]

        if self.closure == 'independent':
            self.member.beta_closure = self.member.beta  * self.CM_SI + \
                                  self.member.betap * self.CM_SH
            self.member.yS_holder = self.member.beta_closure * y[iS]
        else:
            self.member.beta_closure = (sps.kron(np.array([self.member.beta, self.member.betap]), self.L).T.dot(y[iI[0]:(iH[-1]+1)])).T
            self.member.yS_holder = self.member.beta_closure  * y[iS]

        self.member.y_dot     =   self.member.coeffs.dot(y) + self.member.offset
        self.member.y_dot[iS] = - self.member.yS_holder

        return self.member.y_dot

    def integrator(self, closure, start_time, stop_time, y0, maxdt, method='RK45'):
        """
        Function called for ode integration of master equation.
        """
        mkl.set_num_threads(1)
        result = solve_ivp(
                fun = self.compute_rhs,
                t_span = [start_time, stop_time],
                y0 = y0,
                t_eval = [stop_time],
                method = method,
                max_step = maxdt)
        return result
