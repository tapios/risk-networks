import os, sys; sys.path.append(os.path.join(".."))

import numpy as np

from epiforecast.observations import FullStateObservation
from epiforecast.data_assimilation import DataAssimilator

population = 100
parameters = np.zeros(population)

assimilator = DataAssimilator(observations = FullStateObservation(population), 
                                parameters = parameters,
                                    errors = [])
