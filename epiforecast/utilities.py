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
