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

def complement_mask(
        indices,
        array_size):
    """
    Get mask of complement `indices` in 0..(array_size-1)

    Input:
        indices (list),
                (np.array): either of the two:
            - boolean array of size `array_size`
            - array with indices, each of which is in 0..(array_size-1)
    Output:
        mask (np.array): boolean array of complement indices
    """
    mask = np.ones(array_size, dtype=bool)
    mask[indices] = False
    return mask

def mask_by_compartment(
        states,
        compartment):
    """
    Get mask of indices for which state is equal to `compartment`

    Input:
        states (dict): a mapping node -> state
        compartment (char): which compartment to return mask for
    Output:
        mask (np.array): boolean array of indices
    """
    states_array = np.fromiter(states.values(), dtype='<U1')
    mask = (states_array == compartment)
    return mask


