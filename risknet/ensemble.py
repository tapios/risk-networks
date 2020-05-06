import EoN
import numpy as np
import networkx as nx
import random

from scipy import integrate
import scipy.sparse as sps
from collections import defaultdict
from tqdm.autonotebook import tqdm

import epimodels

class epiens(object):

	def __init__(self, M, G, N):
		self.ensemble = [epimodels.static(G, N) for m in range(M)]
		self.M = M
		self.N = N

	def init(self, **kwargs):
		"""
			priors (dict):
		"""

		self.beta  = 0.06
		self.betap = 0.0001 * self.beta

		if kwargs.get('hom', True):
			self.__heterogoneous = False
			[member.init(beta = kwargs.get('beta', 0.06)) for member in self.ensemble]

			self.sigma, self.delta, self.theta, self.thetap, self.mu, self.mup, self.gamma, self.gammap = np.empty(shape = (8, self.M, self.N))

			for mm, member in enumerate(self.ensemble):
				self.sigma[mm,:]  =  member.sigma * np.random.uniform(.7, 1.3, size = self.N)
				self.delta[mm,:]  =  member.delta * np.random.uniform(.7, 1.3, size = self.N)
				self.theta[mm,:]  =  member.gamma * np.random.uniform(.7, 1.3, size = self.N)
				self.thetap[mm,:] =  member.gammap * np.random.uniform(.7, 1.3, size = self.N)
				self.mu[mm,:]     =  member.mu * np.random.uniform(.7, 1.3, size = self.N)
				self.mup[mm,:]    =  member.mup * np.random.uniform(.7, 1.3, size = self.N)
				self.gamma[mm,:]  =  (member.gamma + member.delta + member.mu) * np.random.uniform(.7, 1.3, size = self.N)
				self.gammap[mm,:] =  (member.gammap + member.mup) * np.random.uniform(.7, 1.3, size = self.N)

		else:
			self.__heterogoneous = True

			pass

	def set_solver(self, method = 'RK45', T = 200, dt = 0.1):
		self.method = method
		self.dt = dt
		self.T = T
		self.solve_init = True

		self.set_parameters()

	def set_parameters(self):
		"""
		Set and initialize master equation parameters
		coeffs (array): an array of sparse matrices:
		L (sparse matrix): adjacency Matrix.
		PM (matrix): ensemble centering matrix (M times M).
		"""
		self.coeffs = np.empty(self.M, dtype = 'object')

		for mm in range(self.M):
			self.coeffs[mm] = sps.csr_matrix(sps.bmat(
			[
			   [-sps.eye(self.N), None, None, None, None, None],
			   [ sps.eye(self.N), sps.diags(-self.sigma[mm]), None, None, None, None],
			   [None, sps.diags(self.sigma[mm]), sps.diags(-self.gamma[mm]), None, None, None],
			   [None, None, sps.diags(self.delta[mm]), sps.diags(-self.gammap[mm]), None, None],
			   [None, None, sps.diags(self.theta[mm]), sps.diags(self.thetap[mm]), None, None],
			   [None, None, sps.diags(self.mu[mm]), sps.diags(self.mup[mm]), None, None]
			], format = 'csr'), shape = [6 * self.N, 6 * self.N])

		self.L = nx.to_scipy_sparse_matrix(self.ensemble[0].G)
		self.PM = np.identity(self.M) - 1./self.M * np.ones([self.M,self.M])

	def ens_keqns_sparse(self, t, y):
		"""
		Inputs:
		y (array): an array of dims (N_statuses, N_nodes)
		t (array): times for ode solver

		Returns:
		y_dot (array): lhs of master eqns
		"""
		# Initial conditions: y
		# Initial conditions for all ensemble members Y0 = [y0, ..., y0]
		Y_dot = np.zeros([self.M, 6 * self.N])

		Sidx, Eidx, Iidx, Hidx, Ridx, Didx = [range(jj * self.N, (jj + 1) * self.N) for jj in range(6)]
		self.beta_closure_ind = sps.kron(np.array([self.beta, self.betap]), self.L).dot(
									(y.reshape(self.M, 6 * self.N))[:,[Iidx,Hidx]].reshape(self.M,-1).T
									)

		# Matrix vector multiply for each
		for mm, coeffs in enumerate(self.coeffs):
			coeffs[Sidx,Sidx] = - self.beta_closure_ind[:,mm]
			coeffs[Eidx,Sidx] =   self.beta_closure_ind[:,mm]
			Y_dot[mm] = coeffs.dot(y.reshape(self.M,6*self.N)[mm])

		return Y_dot.flatten()

	def ens_keqns_sparse_closure(self, t, y):
		"""
		Inputs:
		y (array): an array of dims (M times N_statuses times N_nodes)
		t (array): times for ode solver

		Returns:
		y_dot (array): lhs of master eqns
		"""
		# Y = [y_1, .., y_M]
		y0 = np.copy(y)
		Y_dot = np.zeros([self.M, 6 * self.N])

		Sidx, Eidx, Iidx, Hidx, Ridx, Didx = [range(jj * self.N, (jj + 1) * self.N) for jj in range(6)]

		self.beta_closure_ind = sps.kron(np.array([self.beta, self.betap]), self.L).dot(
									(y.reshape(self.M, 6 * self.N))[:,[Iidx,Hidx]].reshape(self.M,-1).T
									)
		self.beta_closure_ens = np.sqrt(1/(self.M - 1)) * np.asarray(self.L.multiply(y.reshape(self.M, -1)[:,Sidx].T.dot(self.PM).dot((self.beta * y.reshape(self.M, -1)[:,Iidx].T + self.betap * y.reshape(self.M, -1)[:,Hidx].T).T)).sum(axis = 1))
		y0.reshape(self.M, -1)[:,Sidx] = y0.reshape(self.M, -1)[:,Sidx] * self.beta_closure_ind.T + self.beta_closure_ens.T

		# Matrix vector multiply for each
		for mm, coeffs in enumerate(self.coeffs):
			Y_dot[mm] = coeffs.dot(y0.reshape(self.M,6*self.N)[mm])

		return Y_dot.flatten()

	def ens_solve(self, y0, t, args = (), **kwargs):
		"""
		"""
		if self.solve_init:
			res = integrate.solve_ivp(
				fun = lambda t, y: self.ens_keqns_sparse(t, y, *args),
				t_span = [0,self.T], y0 = y0,
				t_eval = t, method = self.method, max_step = self.dt)
		else:
			res = np.empty()
		return res

	def ens_solve_closure(self, y0, t, args = (), **kwargs):
		"""
		"""
		if self.solve_init:
			res = integrate.solve_ivp(
				fun = lambda t, y: self.ens_keqns_sparse_closure(t, y, *args),
				t_span = [0,self.T], y0 = y0,
				t_eval = t, method = self.method, max_step = self.dt)
		else:
			res = np.empty()
		return res
