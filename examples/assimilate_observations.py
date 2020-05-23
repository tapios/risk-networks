import os, sys; sys.path.append(os.path.join(".."))

import numpy as np

from epiforecast.observations import FullObservation
from epiforecast.data_assimilator import DataAssimilator

population = 100
n_status = 5
noise_var = 0.01 # independent variance on observations
n_samples = 2 #minimum number for an 'ensemble'

assimilator = DataAssimilator(observations = FullObservation(population,noise_var,"Full state observation 1% noise"), 
                                    errors = [])


#current state, transition_rates (20 of them), transmission rate (1)
current_state = np.random.uniform(0.0, 1.0, (n_samples, population * n_status))
transition_rates = np.random.uniform(0.01, 0.5, (n_samples,100))
transmission_rates = np.random.uniform(0.01, 0.5, (n_samples,1))

#some data (corresponding to the size of the current state)
synthetic_data = 1.0/6.0 * np.ones(population * n_status)

params=np.hstack([transition_rates,transmission_rates])
next_state,next_params=assimilator.update(current_state,params,synthetic_data)

