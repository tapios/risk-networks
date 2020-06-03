import numpy as np
from numba import njit

@njit
def random_printing(n):
    for i in range(n):
        print(np.random.randint(0, high=10))

@njit
def set_seed(seed):
    np.random.seed(seed)

if __name__ == '__main__':

    set_seed(123)
    random_printing(4)
