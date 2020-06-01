import os, sys; sys.path.append(os.path.join(".."))


from epiforecast.health_service import HealthService
from epiforecast.kinetic_model_simulator import print_statuses


import numpy as np
import networkx as nx

def random_infection(population, initial_hospitalized):
    """
    Returns a status dictionary associated with a random infection
    """
    statuses = {node: 'S' for node in range(population)}
    
    initial_hospitalized_nodes = np.random.choice(population, size=initial_hospitalized, replace=False)

    for i in initial_hospitalized_nodes:
        statuses[i] = 'H'

    return statuses


np.random.seed(91210)

population = 100
attachment = 2
network = nx.barabasi_albert_graph(population, attachment)
statuses = random_infection(100,10)

patient_capacity = 4
health_worker_population = 8


health_service = HealthService(patient_capacity,
                               health_worker_population,
                               network)

health_service.discharge_and_admit_patients(statuses,network)

print_statuses(statuses)




