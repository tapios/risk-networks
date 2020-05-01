import EoN
import numpy as np
import networkx as nx
from collections import defaultdict
import random
from tqdm.autonotebook import tqdm

class epinet(object):

	def __init__(self, N, G):
		self.N = N
		self.G = G

class static(epinet):

	def __init__(self, N, G):
		super().__init__(N, G)

	def create_spontaneous(self, sigma = 1/3.5, gamma = 1/13.7):
		mu, delta = np.linalg.solve(np.array([[1 - 0.01,  -0.01],[-0.15, 1 - 0.15]]),
									np.array([[0.01 *(gamma)],[0.15 *(gamma)]]))

		gammap = 1/(1/gamma + 7.0)
		mup    = (0.1/(1-.1)) * (1/(gamma + 7.0))

		self.H = nx.DiGraph()
		self.H.add_node('S')
		self.H.add_edge('E', 'I', rate = sigma)
		self.H.add_edge('I', 'R', rate = gamma)
		self.H.add_edge('H', 'R', rate = gammap)
		self.H.add_edge('I', 'H', rate = delta[0])
		self.H.add_edge('I', 'D', rate = mu[0])
		self.H.add_edge('H', 'D', rate = mup)

	def create_induced(self, beta = 0.06):
		self.J = nx.DiGraph()
		self.J.add_edge(('I', 'S'), ('I', 'E'), rate = beta)
		self.J.add_edge(('H', 'S'), ('H', 'E'), rate = beta)

	def initialize_infected(self, nodes):
		self.IC = defaultdict(lambda: 'S')
		for node in nodes:
			IC[node] = 'I'

	def net_init(self, sigma = 1/3.5, gamma = 1/13.7, beta = 0.06):
		self.create_spontaneous(sigma, gamma)
		self.create_induced(beta)

	def simulate(self, **kwargs):
		simulation = EoN.Gillespie_simple_contagion(
						self.G,
						self.H,
						self.J,
						self.IC,
						return_full_data = True,
						tmax = float('Inf')
					)
		return simulation

class dynamic(epinet):

	def __init__(self, N, G):
		super().__init__(N, G)
