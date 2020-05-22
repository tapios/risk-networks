import numpy as np
from tqdm.autonotebook import tqdm

import sys
sys.path.append('../risknet')

import epitools

import matplotlib.pyplot as plt
import matplotlib

import pickle
import pdb

T = 10.
t = np.linspace(0,T,41)
x_all = pickle.load(open('data/x.pkl', 'rb'))

matplotlib.rcParams.update({'font.size':15})
epitools.plot_eon_ode_only(x_all, t[:-1], xlims = (-.25, 10), alpha = .2)
plt.tight_layout()
plt.savefig('ensemble_EAKF.png')
plt.show()
