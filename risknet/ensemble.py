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
		if isinstance(G, list):
			self.ensemble = [epimodels.static(Gmm, N) for Gmm in G]
		else:
			self.ensemble = [epimodels.static(G, N) for m in range(M)]
		self.M = M
		self.N = N

	def init(self, beta = 0.06, **kwargs):
		"""
			priors (dict):
		"""

		self.beta  = beta
		self.betap = 0.75 * beta

		if kwargs.get('hom', True):
			self.ix_heterogeneous = False
			[member.init(beta = kwargs.get('beta', self.beta)) for member in self.ensemble]

			for member in self.ensemble:
				# Homogeneous parametrization (old, only used for prototyping):
				member.sigma  =  member.sigma * np.random.uniform(0.9, 1.1, size = self.N)   # σ
				member.delta  =  member.delta * np.random.uniform(0.9, 1.1, size = self.N)   # h γ
				member.theta  =  member.gamma * np.random.uniform(0.9, 1.1, size = self.N)   # (1 - d - h) γ
				member.thetap =  member.gammap * np.random.uniform(0.9, 1.1, size = self.N)  # (1 - d') γ'
				member.mu     =  member.mu * np.random.uniform(0.9, 1.1, size = self.N)      # d  γ
				member.mup    =  member.mup * np.random.uniform(0.9, 1.1, size = self.N)     # d' γ'
				member.gamma  =  (member.theta + member.delta + member.mu)                   # γ
				member.gammap =  (member.thetap + member.mup)                                # γ'
		else:
			self.ix_heterogeneous = True
			self.beta  = beta
			self.betap = 0.75 * beta

			for member in self.ensemble:
				member.init(beta = kwargs.get('beta', self.beta), **kwargs)
				member.betap = 0.75 * member.beta

				member.sigma = np.array(list(nx.get_node_attributes(member.G, 'sigma').values()))
				member.delta = np.array(list(nx.get_node_attributes(member.G, 'delta').values()))
				member.theta = np.array(list(nx.get_node_attributes(member.G, 'theta').values()))
				member.thetap = np.array(list(nx.get_node_attributes(member.G, 'thetap').values()))
				member.mu = np.array(list(nx.get_node_attributes(member.G, 'mu').values()))
				member.mup = np.array(list(nx.get_node_attributes(member.G, 'mup').values()))
				member.gamma = np.array(list(nx.get_node_attributes(member.G, 'gamma').values()))
				member.gammap = np.array(list(nx.get_node_attributes(member.G, 'gammap').values()))

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
			member.y_dot = np.zeros(6 * self.N,)

		self.ix_reduced = False
		self.PM = np.identity(self.M) - 1./self.M * np.ones([self.M,self.M])

	def set_parameters_reduced(self):
			"""
			Set and initialize master equation parameters
			coeffs (array): an array of sparse matrices:
			L (sparse matrix): adjacency Matrix.
			PM (matrix): ensemble centering matrix (M times M), idempotent, symmetric.
			"""
			iS, iI, iH = [range(jj * self.N, (jj + 1) * self.N) for jj in range(3)]
			for member in self.ensemble:
					member.coeffs = sps.csr_matrix(sps.bmat(
					[
							[-sps.eye(self.N), None, None, None, None],
							[sps.diags(-member.sigma), sps.diags(- member.sigma - member.gamma), sps.diags(-member.sigma),sps.diags(-member.sigma),sps.diags(-member.sigma)],
							[None, sps.diags(member.delta), sps.diags(-member.gammap), None, None],
							[None, sps.diags(member.theta), sps.diags(member.thetap), None, None],
							[None, sps.diags(member.mu), sps.diags(member.mup), None, None]
					], format = 'csr'), shape = [5 * self.N, 5 * self.N])
					member.offset = np.zeros(5 * self.N,)
					member.offset[iI] = member.sigma
					member.y_dot = np.zeros_like(5 * self.N,)
					member.L = nx.to_scipy_sparse_matrix(member.G)

			self.ix_reduced = True
			self.PM = np.identity(self.M) - 1./self.M * np.ones([self.M,self.M])

	def ens_keqns_sparse_closure(self, t, y, member, member_id, **kwargs):
		"""
		Inputs:
		y (array): an array of dims (M times N_statuses times N_nodes)
		t (array): times for ode solver

		Returns:
		y_dot (array): lhs of master eqns
		"""
		member.y0 = np.copy(y)

		iS, iE, iI, iH = [range(jj * member.N, (jj + 1) * member.N) for jj in range(4)]

		# This makes the update:
		# Y[S] = Y[S] * beta_closure_ind + beta_closure_cov
		# where beta_closure_ind is based on Y[SI] = Y[S]·Y[I]
		member.beta_closure_ind = sps.kron(np.array([member.beta, member.betap]), member.L).dot(member.y0[iI[0]:(iH[-1]+1)])

		if kwargs.get('closure', 'individual') == 'covariance':
			member.y0[iS] = member.beta_closure_ind * member.y0[iS] + self.beta_closure_cov
		elif kwargs.get('closure', 'individual') == 'correlation':
			member.y0[iS] = member.beta_closure_ind * member.y0[iS] + \
								self.beta_closure_cor[:, member_id]
		else:
			member.y0[iS] = member.beta_closure_ind * member.y0[iS]

		member.y_dot = member.coeffs.dot(member.y0)
		member.y_dot[  (member.y_dot > member.y0/self.dt)   & ((member.y_dot < 0) &  (member.y0 < 1e-12)) ] = 0.
		member.y_dot[(member.y_dot < (1-member.y0)/self.dt) & ((member.y_dot > 0) & (member.y0 > 1-1e-12))] = 0.

		return member.y_dot

	def ens_keqns_sparse_closure_reduced(self, t, y, member, member_id, **kwargs):
		"""
		Inputs:
		y (array): an array of dims (M times N_statuses times N_nodes)
		t (array): times for ode solver

		Returns:
		y_dot (array): lhs of master eqns
		"""
		iS, iI, iH = [range(jj * member.N, (jj + 1) * member.N) for jj in range(3)]

		# This makes the update:
		# Y[S] = Y[S] * beta_closure_ind + beta_closure_cov
		# where beta_closure_ind is based on Y[SI] = Y[S]·Y[I]
		# 									 Y[SH] = Y[S]·Y[H]
		member.beta_closure_ind = sps.kron(np.array([member.beta, member.betap]), member.L).dot(y[iI[0]:(iH[-1]+1)])

		if kwargs.get('closure', 'individual') == 'covariance':
			member.yS_holder = member.beta_closure_ind * y[iS] + self.beta_closure_cov
		elif kwargs.get('closure', 'individual') == 'correlation':
			member.yS_holder = member.beta_closure_ind * y[iS] + \
				self.beta_closure_cor[:, member_id]
		elif kwargs.get('closure', 'individual') == 'independent':
			member.yS_holder = self.beta_closure_indp[:, member_id] * y[iS]
		else:
			member.yS_holder = member.beta_closure_ind * y[iS]

		member.y_dot     = member.coeffs.dot(y) + member.offset
		member.y_dot[iS] = - member.yS_holder
		# member.y_dot[  (member.y_dot > y/self.dt)   & ((member.y_dot < 0) &  (y < 1e-12)) ] = 0.
		# member.y_dot[(member.y_dot < (1-y)/self.dt) & ((member.y_dot > 0) & (y > 1-1e-12))] = 0.
		# member.y_dot[y < 1e-8 ] = 0.0

		return member.y_dot

	def eval_closure(self, y, **kwargs):
		if self.ix_reduced:
			iS, iI, iH = [range(jj * self.N, (jj + 1) * self.N) for jj in range(3)]
		else:
			iS, iE, iI, iH = [range(jj * self.N, (jj + 1) * self.N) for jj in range(4)]

		self.L = self.ensemble[0].L

		if kwargs.get('closure', 'individual') == 'covariance':
			# This should be read as:
			# Y[S] PM ( \beta Y[I] + \beta' Y[H]))^T/(M-1) = Cov(Y[S], \beta Y[I] + \beta' Y[H]),
			# Then we apply the Hadamard product with L (the contact matrix):
			# L • Cov(Y[S], \beta Y[I] + \beta' Y[H])
			self.beta_closure_cov = (1/(self.M-1)) * np.asarray(self.L.multiply(
				y[:,iS].T.dot(self.PM).dot((self.beta * y[:,iI].T + self.betap * y[:,iH].T).T)).sum(axis = 1)
			).flatten()

		elif kwargs.get('closure', 'individual') == 'correlation':
			# Might not be optimal, but at least the equations are readable
			self.covSI = y[:,iS].T.dot(self.PM).dot(y[:,iI].T.dot(self.PM).T)/(self.M-1)
			self.varS = np.sqrt((y[:,iS].T.dot(self.PM) * y[:,iS].T.dot(self.PM)).sum(axis = 1)/(self.M-1))
			self.varI = np.sqrt((y[:,iI].T.dot(self.PM) * y[:,iI].T.dot(self.PM)).sum(axis = 1)/(self.M-1))
			self.denSI = self.varS.reshape(-1,1).dot(self.varI.reshape(1,-1)) + 1e-8
			self.LcorSI = self.L.multiply(self.covSI/self.denSI)
			self.LcorSI.data[np.isnan(self.LcorSI.data)] = 0.

			self.covSH = y[:,iS].T.dot(self.PM).dot(y[:,iH].T.dot(self.PM).T)/(self.M-1)
			self.varH = np.sqrt((y[:,iH].T.dot(self.PM) * y[:,iH].T.dot(self.PM)).sum(axis = 1)/(self.M-1))
			self.denSH = self.varS.reshape(-1,1).dot(self.varH.reshape(1,-1)) + 1e-8
			self.LcorSH = self.L.multiply(self.covSH/self.denSH)
			self.LcorSH.data[np.isnan(self.LcorSH.data)] = 0.

			self.beta_closure_cor = ((self.beta * self.LcorSI.dot(np.sqrt(y[:,iI] * (1-y[:,iI])).T) + \
									 self.betap * self.LcorSH.dot(np.sqrt(y[:,iH] * (1-y[:,iH])).T)) * \
										   np.sqrt(y[:,iS] * (1-y[:,iS])).T)

		elif kwargs.get('closure', 'individual') == 'independent':
			jSI = y[:,iS].T.dot(y[:,iI])/(self.M)
			iSI = y[:,iS].mean(axis = 0).reshape(-1,1).dot(y[:,iI].mean(axis = 0).reshape(1,-1))

			jSH = y[:,iS].T.dot(y[:,iH])/(self.M)
			iSH = y[:,iS].mean(axis = 0).reshape(-1,1).dot(y[:,iH].mean(axis = 0).reshape(1,-1))
			self.beta_closure_indp = self.beta * self.L.multiply((jSI/(iSI+1e-8))).dot(y[:,iI].T) + \
									 self.betap * self.L.multiply((jSH/(iSH+1e-8))).dot(y[:,iH].T)

	def set_solver(self, method = 'RK45', T = 200, dt = 0.1, reduced = False):

		for member in self.ensemble:
			member.set_solver(method = method, T = T, dt = dt, member_call = False)

		# Ensemble summary
		self.method = method
		self.dt = dt
		self.T = T
		self.solve_init = True

		if reduced:
			self.set_parameters_reduced()
		else:
			self.set_parameters()

	def ens_solve(self, y0, t, args = (), **kwargs):
		"""
		"""
		results = []
		if self.solve_init:
			for mm, member in tqdm(enumerate(self.ensemble), desc = 'Solving member', total = self.M):
				res = member.solve(y0[mm], t, **kwargs)
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
				if self.ix_reduced:
					self.y0[mm] += self.dt * self.ens_keqns_sparse_closure_reduced(t, self.y0[mm], member, mm, **kwargs)
				else:
					self.y0[mm] += self.dt * self.ens_keqns_sparse_closure(t, self.y0[mm], member, mm, **kwargs)
				self.y0[mm] = np.clip(self.y0[mm], 0., 1.)
			self.tf += self.dt
			yt[:,jj + 1] = np.copy(self.y0.flatten())

		return yt.reshape(self.M, -1, len(t))
