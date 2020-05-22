import EoN
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
import random

import matplotlib.pyplot as plt
import matplotlib
import scipy.sparse as sps
import seaborn as sns

from tqdm.autonotebook import tqdm
import time

import sys
sys.path.append('../risknet')

#import epimodels
import ensemble
import epitools

import pickle
from utilities import *
import pdb

# Setting the time
T = 40.
t = np.linspace(0,T,161)

# Load city-like network
print("Loading the network")
start_time = time.time()
G = load_G()
elapsed_time = time.time() - start_time
print("Elapsed time for loading the network: ", elapsed_time)

N = len(G)
model_n_samples = 100 

print("Setting the model")
start_time = time.time()
Gs = model_settings(model_n_samples, G)

# Output the prior of latent period
sigma_ensemble = np.zeros((model_n_samples,N))
iterN = 0
for G in Gs:
    sigma_ensemble[iterN,:] = np.array(list(nx.get_node_attributes(Gs[iterN], 'sigma').values()))
    iterN = iterN + 1
pickle.dump(sigma_ensemble, open("sigma_ens.pkl", "wb"))

model = get_model(Gs, model_n_samples, N)
model.init(beta = 0.06, hom = False)
y0 = get_IC(model, model_n_samples, N)
model.set_solver(T = T, dt = np.diff(t).min(), reduced = True)
elapsed_time = time.time() - start_time
print("Elapsed time for setting the model: ", elapsed_time)

print("Solving the ODEs")
start_time = time.time()
ke_euler = model.ens_solve_euler(y0, t)
pickle.dump(ke_euler, open("states_truth_citynet.pkl", "wb"))
elapsed_time = time.time() - start_time
print("Elapsed time for solving ODEs: ", elapsed_time)

matplotlib.rcParams.update({'font.size':15})
epitools.plot_ode(ke_euler, t, alpha = .2)
plt.tight_layout()
plt.savefig('ensemble_truth.png')
plt.show()
