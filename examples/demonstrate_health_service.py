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

    return statuses, initial_hospitalized_nodes


np.random.seed(91210)

population = 1000
attachment = 2
network = nx.barabasi_albert_graph(population, attachment)
statuses, initial_hospitalized_nodes = random_infection(population,50)

print("Initial statuses")
print_statuses(statuses)
patient_capacity = 15
health_worker_population = 10


health_service = HealthService(patient_capacity,
                               health_worker_population,
                               network)

health_service.discharge_and_admit_patients(statuses,network)

print("Statuses after discharge and admittance")
print_statuses(statuses)

recovered_patients = [patient.address for patient in health_service.patients[:10]]
print("Assume of those hospitalized, 10 patients have recovered",recovered_patients)
statuses.update({node: 'R' for node in recovered_patients})

deceased_community = [health_service.waiting_list[i] for i in np.arange(10)]
print("Assume of those unable to be hospitalized, 10 patients are now deceased",deceased_community)
statuses.update({node: 'D' for node in deceased_community})

new_patients =  [health_service.waiting_list[i] for i in np.arange(10)+10]
print("Assume, of those unable to be hospitalized, 10 patients require hospitalization once more",new_patients)
statuses.update({node: 'H' for node in new_patients})


print("Statuses after events")
print_statuses(statuses)
health_service.discharge_and_admit_patients(statuses,network)


print("Statuses after discharge and admittance")
print_statuses(statuses)



