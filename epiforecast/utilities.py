import numpy as np
import random
from numba import njit

# Utilities for seeding random number generators

@njit
def seed_numba_random_state(seed):
    np.random.seed(seed)

def seed_three_random_states(seed):
    random.seed(seed)
    np.random.seed(seed)
    seed_numba_random_state(seed)

def not_involving(nodes):
    """
    Filters edges that connect to `nodes`.
    """
    def edge_doesnt_involve_any_nodes(edge, nodes=nodes):
        return edge[0] not in nodes and edge[1] not in nodes

    return edge_doesnt_involve_any_nodes

@njit
def normalize(edge):
    """
    Normalize a symmetric edge by placing largest node id first.
    """
    n, m = edge
    if m > n: # switch
        n, m = m, n
    return n, m
