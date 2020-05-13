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

	def init(self, beta = 0.06, **kwargs):
		"""
			priors (dict):
		"""

		self.beta  = beta
		self.betap = 0.0001 * beta

		if kwargs.get('hom', True):
			self.__heterogeneous = False
			[member.init(beta = kwargs.get('beta', self.beta)) for member in self.ensemble]

			for member in self.ensemble:
				# Homogeneous parametrization:
				member.sigma  =  member.sigma * np.random.uniform(0.9, 1.1, size = self.N)
				member.delta  =  member.delta * np.random.uniform(0.9, 1.1, size = self.N)
				member.theta  =  member.gamma * np.random.uniform(0.9, 1.1, size = self.N)
				member.thetap =  member.gammap * np.random.uniform(0.9, 1.1, size = self.N)
				member.mu     =  member.mu * np.random.uniform(0.9, 1.1, size = self.N)
				member.mup    =  member.mup * np.random.uniform(0.9, 1.1, size = self.N)
				member.gamma  =  (member.theta + member.delta + member.mu)
				member.gammap =  (member.thetap + member.mup)

		else:
			self.__heterogeneous = True

			pass

	def set_parameters(self):
		"""
		Set and initialize master equation parameters
		coeffs (array): an array of sparse matrices:
		L (sparse matrix): adjacency Matrix.
		PM (matrix): ensemble centering matrix (M times M), idempotent, symmetric.
		"""

		for member in self.ensemble:
			member.coeffs = sps.csr_matrix(sps.bmat(
			[
				[-sps.eye(self.N), None, None, None, None, None],
				[ sps.eye(self.N), sps.diags(-member.sigma), None, None, None, None],
				[None, sps.diags(member.sigma), sps.diags(-member.gamma), None, None, None],
				[None, None, sps.diags(member.delta), sps.diags(-member.gammap), None, None],
				[None, None, sps.diags(member.theta), sps.diags(member.thetap), None, None],
				[None, None, sps.diags(member.mu), sps.diags(member.mup), None, None]
			], format = 'csr'), shape = [6 * self.N, 6 * self.N])

			member.L = nx.to_scipy_sparse_matrix(member.G)

		self.PM = np.identity(self.M) - 1./self.M * np.ones([self.M,self.M])

	def ens_keqns_sparse(self, t, y, member):
		"""
		Inputs:
		y (array): an array of dims (N_statuses, N_nodes)
		t (array): times for ode solver

		Returns:
		y_dot (array): lhs of master eqns
		"""
		# Initial conditions: y
		# Initial conditions for all ensemble members Y0 = [y0, ..., y0]
		y0 = np.copy(y)

		Sidx, Eidx, Iidx, Hidx = [range(jj * member.N, (jj + 1) * member.N) for jj in range(4)]

		member.beta_closure_ind = sps.kron(np.array([member.beta, member.betap]), member.L).dot(y0[Iidx[0]:(Hidx[-1]+1)])

		y0[Sidx] = member.beta_closure_ind * y0[Sidx]

		return member.coeffs.dot(y0)

	def ens_keqns_sparse_closure(self, t, y, member, member_id, **kwargs):
		"""
		Inputs:
		y (array): an array of dims (M times N_statuses times N_nodes)
		t (array): times for ode solver

		Returns:
		y_dot (array): lhs of master eqns
		"""
		member.y0 = np.copy(y)

		Sidx, Eidx, Iidx, Hidx = [range(jj * member.N, (jj + 1) * member.N) for jj in range(4)]

		# This makes the update:
		# Y[S] = Y[S] * beta_closure_ind + beta_closure_cov
		# where beta_closure_ind is based on Y[SI] = Y[S]·Y[I]
		member.beta_closure_ind = sps.kron(np.array([member.beta, member.betap]), member.L).dot(member.y0[Iidx[0]:(Hidx[-1]+1)])

		if kwargs.get('closure', 'individual') == 'covariance':
			member.y0[Sidx] = member.beta_closure_ind * member.y0[Sidx] + self.beta_closure_cov
		elif kwargs.get('closure', 'individual') == 'correlation':
			member.y0[Sidx] = member.beta_closure_ind * member.y0[Sidx] + \
								self.beta_closure_cor[:, member_id]
		else:
			member.y0[Sidx] = member.beta_closure_ind * member.y0[Sidx]

		return member.coeffs.dot(member.y0)

	def eval_closure(self, y, **kwargs):
		Sidx, Eidx, Iidx, Hidx = [range(jj * self.N, (jj + 1) * self.N) for jj in range(4)]
		self.L = self.ensemble[0].L
		if kwargs.get('closure', 'individual') == 'covariance':
			# This should be read as:
			# Y[S] PM ( \beta Y[I] + \beta' Y[H]))^T/(M-1) = Cov(Y[S], \beta Y[I] + \beta' Y[H]),
			# Then we apply the Hadamard product with L (the contact matrix):
			# L • Cov(Y[S], \beta Y[I] + \beta' Y[H])
			self.beta_closure_cov = (1/(self.M-1)) * np.asarray(self.L.multiply(
				y[:,Sidx].T.dot(self.PM).dot((self.beta * y[:,Iidx].T + self.betap * y[:,Hidx].T).T)).sum(axis = 1)
			).flatten()

		elif kwargs.get('closure', 'individual') == 'correlation':
			# Might not be optimal, but at least the equations are readable
			self.covSI = y[:,Sidx].T.dot(self.PM).dot(y[:,Iidx].T.dot(self.PM).T)/(self.M-1)
			self.varS = np.sqrt((y[:,Sidx].T.dot(self.PM) * y[:,Sidx].T.dot(self.PM)).sum(axis = 1)/(self.M-1))
			self.varI = np.sqrt((y[:,Iidx].T.dot(self.PM) * y[:,Iidx].T.dot(self.PM)).sum(axis = 1)/(self.M-1))
			self.denSI = self.varS.reshape(-1,1).dot(self.varI.reshape(1,-1))
			self.LcorSI = self.L.multiply(self.covSI * (1/self.denSI))
			self.LcorSI.data[np.isnan(self.LcorSI.data)] = 0.

			self.covSH = y[:,Sidx].T.dot(self.PM).dot(y[:,Hidx].T.dot(self.PM).T)/(self.M-1)
			self.varH = np.sqrt((y[:,Hidx].T.dot(self.PM) * y[:,Hidx].T.dot(self.PM)).sum(axis = 1)/(self.M-1))
			self.denSH = self.varS.reshape(-1,1).dot(self.varH.reshape(1,-1))
			self.LcorSH = self.L.multiply(self.covSH * (1/self.denSH))
			self.LcorSH.data[np.isnan(self.LcorSH.data)] = 0.

			#
			self.beta_closure_cor = ((self.beta * self.LcorSI.dot(np.sqrt(y[:,Iidx] * (1-y[:,Iidx])).T) + \
 									 self.betap * self.LcorSH.dot(np.sqrt(y[:,Hidx] * (1-y[:,Hidx])).T)) * \
									       np.sqrt(y[:,Sidx] * (1-y[:,Sidx])).T)

	def set_solver(self, method = 'RK45', T = 200, dt = 0.1):

		for member in self.ensemble:
			member.set_solver(method = method, T = T, dt = dt, member_call = False)

		self.method = method
		self.dt = dt
		self.T = T

		self.solve_init = True
		self.set_parameters()

	def ens_solver_euler(self, **kwargs):
		pass

	def ens_solve(self, y0, t, args = (), **kwargs):
		"""
		"""
		results = []
		if self.solve_init:
			for mm, member in tqdm(enumerate(self.ensemble), desc = 'Solving member', total = self.M):
				res = integrate.solve_ivp(
						fun = lambda t, y: self.ens_keqns_sparse_closure(t, y, member, mm, **kwargs),
						t_span = [0, self.T], y0 = y0[mm],
						t_eval = t, method = self.method, max_step = self.dt)
				results.append(res)
		return results

	def ens_solve_euler(self, y0, t, args = (), **kwargs):
		"""
		"""
		self.tf = 0.
		self.y0 = np.copy(y0)
		yt = np.empty((len(y0.flatten()), len(t)))
		yt[:,0] = np.copy(y0.flatten())

		for jj, time in tqdm(enumerate(t[:-1]), desc = 'Forward pass', total = len(t[:-1])):
			self.eval_closure(self.y0, **kwargs)
			for mm, member in enumerate(self.ensemble):
				self.y0[mm] += self.dt * self.ens_keqns_sparse_closure(t, self.y0[mm], member, mm, **kwargs)
			self.tf += self.dt
			yt[:,jj + 1] = np.copy(self.y0.flatten())

		return yt.reshape(self.M, -1, len(t))
