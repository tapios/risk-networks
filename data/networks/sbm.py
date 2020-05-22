import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
import math
import os
import itertools
import collections
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

edge_list = np.loadtxt('edge_list.txt')

G = nx.Graph()

G.add_edges_from(edge_list)

#c = list(greedy_modularity_communities(G))

#print(c[0])
#print(c[1])
#print(c[2])

G2 = nx.subgraph(G, np.arange(500))

degree_sequence = sorted([d for n, d in G2.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

plt.figure()

plt.plot(deg, cnt, 'o')

plt.xlabel(r'$k$')
plt.ylabel(r'$P(k)$')

plt.tight_layout()

plt.show()

G3 = nx.subgraph(G, np.arange(500,10000))

degree_sequence = sorted([d for n, d in G3.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

xx = np.linspace(10,100)

plt.figure()

plt.plot(xx, 10**5*xx**-2.5, label = r'$P(k)\propto k^{-2.5}$')

plt.plot(deg, cnt, 'o', label = 'SBM')

plt.legend(loc = 1)

plt.xlabel(r'$k$')
plt.ylabel(r'$P(k)$')

plt.xscale('log')
plt.yscale('log')

plt.show()