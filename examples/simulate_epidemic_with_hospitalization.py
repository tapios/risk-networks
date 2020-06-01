import os, sys; sys.path.append(os.path.join(".."))

import networkx as nx
import numpy as np
import random 
import copy
from collections import defaultdict

# Utilities for generating random populations
from epiforecast.populations import populate_ages, sample_distribution, TransitionRates
from epiforecast.samplers import GammaSampler, AgeAwareBetaSampler

from epiforecast.contact_simulator import ContactSimulator, DiurnalMeanContactRate
from epiforecast.kinetic_model_simulator import KineticModel, print_statuses, print_initial_statuses
from epiforecast.scenarios import load_edges

from epiforecast.node_identifier_helper import load_node_identifiers
from epiforecast.health_service import HealthService

def random_infection(population, initial_infected):
    """
    Returns a status dictionary associated with a random infection
    """
    statuses = defaultdict(lambda: 'S') # statuses = np.repeat('S', nodes)

    initial_infected_nodes = np.random.choice(population, size=initial_infected, replace=False)

    for i in initial_infected_nodes:
        statuses[i] = 'I'

    return statuses

#
# Set random seed for reproducibility
#

# Both numpy.random and random are used by the KineticModel.
#np.random.seed(1233) # no error
np.random.seed(123)
random.seed(123)



#
# Load an example network
#

edges = load_edges(os.path.join('..', 'data', 'networks', 'edge_list_SBM_1e3.txt')) 

contact_network = nx.Graph()
contact_network.add_edges_from(edges)
world_population = len(contact_network)

#
# Clinical parameters of an age-distributed population
#

# Now get the clinical parameters for the community and health worker populations.
node_identifiers = load_node_identifiers(os.path.join('..', 'data', 'networks', 'node_identifier_SBM_1e3.txt'))

hospital_bed_number      = node_identifiers["hospital_beds"].size
health_worker_population = node_identifiers["health_workers"].size
community_population     = node_identifiers["community"].size

population = community_population + health_worker_population

#print("Total population:", population)
print("Community population:", community_population)
print("Health worker population:", health_worker_population)
print("Hospital beds:", hospital_bed_number)

# The age category of each community individual,
age_distribution = [ 
                    0.24,   # 0-19 years
                    0.37,   # 20-44 years
                    0.24,   # 45-64 years
                    0.083,  # 65-75 years
                   ]

## 75 onwards
age_distribution.append(1 - sum(age_distribution))

# and the ages of the healthcare individual (of working age)
working_age_distribution= [
                           0.0,                  # 0-19 years
                           0.37 / (0.37 + 0.24), # 20-44 years
                           0.24 / (0.37 + 0.24), # 45-64 years
                           0.0,                  # 65-75 years
                           0.0                   # 75+
                          ]                  
                            
community_ages     = populate_ages(community_population, distribution=age_distribution)
health_worker_ages = populate_ages(health_worker_population, distribution=working_age_distribution)

ages = np.hstack([health_worker_ages, community_ages])

# Next, we randomly generate clinical properties for our example population.
# Note that the units of 'periods' are days, and the units of 'rates' are 1/day.
latent_periods               = sample_distribution(GammaSampler(k=1.7, theta=2.0), population=population, minimum=2)
community_infection_periods  = sample_distribution(GammaSampler(k=1.5, theta=2.0), population=population, minimum=1)
hospital_infection_periods   = sample_distribution(GammaSampler(k=1.5, theta=3.0), population=population, minimum=1)
hospitalization_fraction     = sample_distribution(AgeAwareBetaSampler(mean=[ 0.02,  0.17,  0.25, 0.35, 0.45], b=4), ages=ages)
community_mortality_fraction = sample_distribution(AgeAwareBetaSampler(mean=[0.001, 0.001, 0.005, 0.02, 0.05], b=4), ages=ages)
hospital_mortality_fraction  = sample_distribution(AgeAwareBetaSampler(mean=[0.001, 0.001,  0.01, 0.04,  0.1], b=4), ages=ages)

# We process the clinical data to determine transition rates between each epidemiological state,
transition_rates = TransitionRates(population,
                                   latent_periods,
                                   community_infection_periods,
                                   hospital_infection_periods,
                                   hospitalization_fraction,
                                   community_mortality_fraction,
                                   hospital_mortality_fraction)

transmission_rate = 0.1
hospital_transmission_reduction = 1/4

#
# Build the contact simulator
#

mean_contact_duration = 45 / 60 / 60 / 24 # 45 seconds in units of days
mean_contact_rate = DiurnalMeanContactRate(maximum=22, minimum=3)

contact_simulator = ContactSimulator(n_contacts = len(edges),
                                     mean_event_duration = mean_contact_duration,
                                     mean_contact_rate = mean_contact_rate,
                                     start_time = -3 / 24, # the simulation starts at midnight
                                    )

# Generate initial contact network with 3 hour simulation
contact_simulator.mean_contact_duration(stop_time = 0.0)

#
# Set up the health service
#

health_service = HealthService(node_identifiers, contact_network, edges)
population_network = health_service.create_population_network()

#
# Build the kinetic model
#

kinetic_model = KineticModel(                contact_network = population_network,
                                            transition_rates = transition_rates,
                                 community_transmission_rate = transmission_rate,
                             hospital_transmission_reduction = hospital_transmission_reduction
                            )

# 
# Seed an infection
#

statuses = random_infection(population, 100)
print("Number of nodes in the initial condition:", len(statuses))
print("Number of nodes in the contact_network:", len(kinetic_model.contact_network))

# for node, status in statuses.items():
#     print("Node:", node, "status:", status, "neighbors:", [n for n in iter(kinetic_model.contact_network[node])])

# for node in kinetic_model.contact_network.nodes():
#     print("Node", node, "in the contact network has status", statuses[node])

print("Structure of the initial infection:")
#print_initial_statuses(statuses,population)

# 
# Simulate the uncontrolled growth of an epidemic for 7 days
#

print("\n *** The uncontrolled growth phase *** \n")

contacts_time_series = []

# Interval over which the contacts are static from the standpoint of
# the KineticModel
static_contact_interval = 3/24 # days
growth_interval = 7 # days

growth_steps = int(growth_interval / static_contact_interval)
growth_start_times = static_contact_interval * (np.arange(growth_steps-1)+1)

# Run the simulation
for i in range(growth_steps):

    interval_averaged_contact_duration = contact_simulator.mean_contact_duration(stop_time = growth_start_times[i])
    contacts_time_series.append(interval_averaged_contact_duration)
        
    print("Simulating the uncontrolled growth phase of an epidemic",
          "from day {:.3f}".format(growth_start_times[i]),
          "until day {:.3f}".format(growth_start_times[i] + static_contact_interval)) 


    #adjust network to account for hospitalizations
    health_service.discharge_and_admit_patients(statuses, interval_averaged_contact_duration, population_network)
    print_statuses(statuses)
    kinetic_model.set_contact_network(population_network)
    
    initial_statuses = copy.deepcopy(statuses)
    statuses = None

    statuses = kinetic_model.simulate(static_contact_interval, initial_statuses=initial_statuses)
 
# 
# Simulate the hopeful death of an epidemic after social distancing
#
exit()
print("\n *** The social distancing phase *** \n")

# Because the epidemic is out of control, we social distance.
contact_simulator.mean_contact_rate = DiurnalMeanContactRate(maximum=5, minimum=3)

distancing_interval = 3 # days
distancing_steps = int(distancing_interval / static_contact_interval)
distancing_start_times = (growth_start_times[-1] 
    + (static_contact_interval) * np.arange(start=1, stop=distancing_steps+1))
                        
# Run the simulation
for i in range(distancing_steps):

    print("Simulating the social distancing phase of an epidemic",
          "from day {:.3f}".format(distancing_start_times[i]),
          "until day {:.3f}".format(distancing_start_times[i] + static_contact_interval))

    contact_simulator.simulate_contact_duration(stop_time = distancing_start_times[i])
    interval_averaged_contact_duration = contact_simulator.contact_duration / static_contact_interval
    contacts_time_series.append(interval_averaged_contact_duration)
    
    kinetic_model.set_contact_network(population_network)
    
    # Based on the current 'node statuses'
    health_service.discharge_and_obtain_patients(statuses)
    
    statuses = kinetic_model.simulate(static_contact_interval)
