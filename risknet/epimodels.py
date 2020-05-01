import EoN
import numpy as np
import networkx as nx
import random

from scipy import integrate
from collections import defaultdict
from tqdm.autonotebook import tqdm

class epinet(object):

	def __init__(self, G, N):
		self.N = N
		self.G = G

class static(epinet):

	def __init__(self, G, N):
		super().__init__(G, N)

	def create_spontaneous(self, sigma = 1/3.5, gamma = 1/13.7):
		self.sigma = sigma
		self.gamma = gamma
		mu, delta = np.linalg.solve(np.array([[1 - 0.01,  -0.01],[-0.15, 1 - 0.15]]),
									np.array([[0.01 *(self.gamma)],[0.15 *(self.gamma)]]))
		self.mu = mu[0]
		self.delta = delta[0]

		self.gammap = 1/(1/self.gamma + 7.0)
		self.mup    = (0.1/(1-.1)) * (1/(self.gamma + 7.0))

		self.H = nx.DiGraph()
		self.H.add_node('S')
		self.H.add_edge('E', 'I', rate = self.sigma)
		self.H.add_edge('I', 'R', rate = self.gamma)
		self.H.add_edge('H', 'R', rate = self.gammap)
		self.H.add_edge('I', 'H', rate = self.delta)
		self.H.add_edge('I', 'D', rate = self.mu)
		self.H.add_edge('H', 'D', rate = self.mup)

	def create_induced(self, beta = 0.06):
		self.beta  = beta
		self.betap = beta

		self.J = nx.DiGraph()
		self.J.add_edge(('I', 'S'), ('I', 'E'), rate = self.beta)
		self.J.add_edge(('H', 'S'), ('H', 'E'), rate = self.betap)

	def init_infected(self, nodes):
		self.IC = defaultdict(lambda: 'S')
		for node in nodes:
			self.IC[node] = 'I'

	def init(self, sigma = 1/3.5, gamma = 1/13.7, beta = 0.06):
		self.create_spontaneous(sigma, gamma)
		self.create_induced(beta)

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

	def kolmogorov_eqns_ind(self, t, y):
		"""
		Inputs:
		y (array): an array of dims (N_statuses, N_nodes)
		t (array): times for ode solver

		Returns:
		y_dot (array): lhs of master eqns
		"""
		S, E, I, H, R, D = y.reshape(6, self.N)
		S_dot, E_dot, I_dot, H_dot, R_dot, D_dot = np.zeros([6, self.N])

		for ii, ngbr in self.G.adjacency():
			jj = list(ngbr.keys())
			S_dot[ii] = -(self.beta * I[jj] + self.betap * H[jj]).sum() * S[ii]
			E_dot[ii] =  (self.beta * I[jj] + self.betap * H[jj]).sum() * S[ii] - self.sigma * E[ii]

		I_dot = self.sigma * E - (self.gamma + self.delta + self.mu) * I
		H_dot = self.delta * I - (self.gammap + self.mup) * H
		R_dot = self.gamma * I + self.gammap * H
		D_dot = self.mu * I + self.mup * H

		return np.hstack((S_dot, E_dot, I_dot, H_dot, R_dot, D_dot))

	def solve(self, y0, t, args = ()):
		"""
		"""
		if self.solve_init:
			res = integrate.solve_ivp(
				fun = lambda t, y: self.kolmogorov_eqns_ind(t, y, *args),
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
