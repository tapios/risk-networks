import EoN
import numpy as np
import networkx as nx
import random

from scipy import integrate
import scipy.sparse as sps
from collections import defaultdict
from tqdm.autonotebook import tqdm

class epinet(object):

	def __init__(self, G, N):
		self.N = N
		self.G = G

class static(epinet):

	def __init__(self, G, N):
		super().__init__(G, N)

	def create_spontaneous(self, sigma = 1/3.5, gamma = 1/13.7, **kwargs):

		self.H = nx.DiGraph()
		self.H.add_node('S')

		if self.__heterogoneous:

			self.H.add_edge('E', 'I', rate = 1., weight_label = 'sigma')

			self.H.add_edge('I', 'R', rate = 1., weight_label = 'theta')
			self.H.add_edge('I', 'H', rate = 1., weight_label = 'delta')
			self.H.add_edge('I', 'D', rate = 1., weight_label = 'mu')

			self.H.add_edge('H', 'R', rate = 1., weight_label = 'thetap')
			self.H.add_edge('H', 'D', rate = 1., weight_label = 'mup')

		else:
			self.sigma = sigma
			self.gamma = gamma
			mu, delta = np.linalg.solve(np.array([[1 - 0.01,  -0.01],[-0.15, 1 - 0.15]]),
										np.array([[0.01 *(self.gamma)],[0.15 *(self.gamma)]]))
			self.mu = mu[0]
			self.delta = delta[0]

			self.gammap = 1/(1/self.gamma + 7.0)
			self.mup    = (0.1/(1-.1)) * (1/(self.gamma + 7.0))

			self.H.add_edge('E', 'I', rate = self.sigma)

			self.H.add_edge('I', 'R', rate = self.gamma)
			self.H.add_edge('I', 'H', rate = self.delta)
			self.H.add_edge('I', 'D', rate = self.mu)

			self.H.add_edge('H', 'R', rate = self.gammap)
			self.H.add_edge('H', 'D', rate = self.mup)

	def create_induced(self, beta = 0.06, **kwargs):
		self.beta  = beta
		self.betap = 0.0001 * beta

		self.J = nx.DiGraph()
		self.J.add_edge(('I', 'S'), ('I', 'E'), rate = self.beta)
		self.J.add_edge(('H', 'S'), ('H', 'E'), rate = self.betap)

	def init_infected(self, nodes):
		self.IC = defaultdict(lambda: 'S')
		for node in nodes:
			self.IC[node] = 'I'

	def init(self, sigma = 1/3.5, gamma = 1/13.7, beta = 0.06, **kwargs):
		if kwargs.get('het', False):
			self.__heterogoneous = True
		else:
			self.__heterogoneous = False

		self.create_spontaneous(sigma, gamma, **kwargs)
		self.create_induced(beta, **kwargs)

	def simulate(self, return_statuses, **kwargs):
		simulation = EoN.Gillespie_simple_contagion(
						self.G,
						self.H,
						self.J,
						self.IC,
						return_statuses = return_statuses,
						return_full_data = True,
						tmax = float('Inf')
					)
		return simulation

	def set_solver(self, method = 'RK45', T = 200, dt = 0.1):
		self.method = method
		self.dt = dt
		self.T = T
		self.solve_init = True

		self.set_parameters()

	def set_parameters(self):
		"""
		Set and initialize master equation parameters
		"""
		if self.__heterogoneous:
			self.sigma = np.array(list(nx.get_node_attributes(self.G, 'sigma').values()))
			self.gamma = np.array(list(nx.get_node_attributes(self.G, 'gamma').values()))
			self.gammap = np.array(list(nx.get_node_attributes(self.G, 'gammap').values()))
			self.theta = np.array(list(nx.get_node_attributes(self.G, 'theta').values()))
			self.delta = np.array(list(nx.get_node_attributes(self.G, 'delta').values()))
			self.mu = np.array(list(nx.get_node_attributes(self.G, 'mu').values()))
			self.thetap = np.array(list(nx.get_node_attributes(self.G, 'thetap').values()))
			self.mup = np.array(list(nx.get_node_attributes(self.G, 'mup').values()))
		else:
			self.sigma =  self.sigma * np.ones(self.N)
			self.delta =  self.delta * np.ones(self.N)
			self.theta =  self.gamma * np.ones(self.N)
			self.thetap =  self.gammap * np.ones(self.N)
			self.mu =  self.mu * np.ones(self.N)
			self.mup =  self.mup * np.ones(self.N)
			self.gamma =  (self.gamma + self.delta + self.mu) * np.ones(self.N)
			self.gammap =  (self.gammap + self.mup) * np.ones(self.N)


		self.coeffs = sps.csr_matrix(sps.bmat([[sps.eye(self.N), None, None, None, None, None],
		   [sps.eye(self.N), sps.diags(-self.sigma), None, None, None, None],
		   [None, sps.diags(self.sigma), sps.diags(-self.gamma), None, None, None],
		   [None, None, sps.diags(self.delta), sps.diags(-self.gammap), None, None],
		   [None, None, sps.diags(self.theta), sps.diags(self.thetap), None, None],
		   [None, None, sps.diags(self.mu), sps.diags(self.mup), None, None]
		  ], format = 'csr'), shape = [6 * self.N, 6 * self.N])
		self.beta_closure = np.zeros(self.N,)
		self.L = nx.to_scipy_sparse_matrix(self.G)

		self.adj = defaultdict()

		for ii, ngbr in self.G.adjacency():
			self.adj[ii] = list(ngbr.keys())

		self.checks = []

	def kolmogorov_eqns_het_sparse(self, t, y):
		"""
		Inputs:
		y (array): an array of dims (N_statuses, N_nodes)
		t (array): times for ode solver

		Returns:
		y_dot (array): lhs of master eqns
		"""
		S, E, I, H = [range(kk * self.N, (kk + 1) * self.N) for kk in range(4)]

		self.beta_closure = sps.kron(np.array([self.beta, self.betap]),	self.L).dot(np.hstack([y[I], y[H]]))

		self.coeffs[S,S] = - self.beta_closure
		self.coeffs[E,S] =   self.beta_closure

		y = self.coeffs.dot(y)

		return y

	def solve(self, y0, t, args = (), **kwargs):
		"""
		"""
		if self.solve_init:
			res = integrate.solve_ivp(
				fun = lambda t, y: self.kolmogorov_eqns_het_sparse(t, y, *args),
				t_span = [0,self.T], y0 = y0,
				t_eval = t, method = self.method, max_step = self.dt)
		else:
			res = np.empty()
		return res

class dynamic(epinet):

	def __init__(self, N = 0, G = {}):
		super().__init__(N, G)

	def temporal_network(self, edge_list, deltat):
		"""
		temporal network construction

		Parameters:
		edge_list (array): edge list (1st column: time stamp UNIX format, 2nd-3rd columnm: edge i <-> j)
		deltat (float): time step (duration) of each network snapshot in seconds

		Returns:
		Gord: dictionary with time stamps (seconds) and networks

		"""

		G = {}

		G1 = nx.Graph()

		T0 = edge_list[0][0]

		T = edge_list[0][0]

		nodes = edge_list[:,1]
		nodes = np.append(nodes, edge_list[:,2])
		nodes = set(nodes)

		Gnodes = nx.Graph()
		Gnodes.add_nodes_from(nodes)

		for i in range(len(edge_list)):

			if edge_list[i][0] <= T + deltat:
				G1.add_nodes_from(nodes)
				G1.add_edge(edge_list[i][1],edge_list[i][2])
			else:

				if len(G1):
					G[(T-T0)] = G1
					G1 = nx.Graph()
				else:
					G[(T-T0)] = Gnodes
				T += deltat

		Gord = OrderedDict(sorted(G.items()))
		self.G = Gord
