import os, sys; sys.path.append(os.path.join(".."))

import numpy as np
import random 

# Utilities for generating random populations
from epiforecast.populations import populate_ages, sample_clinical_distribution, TransitionRates
from epiforecast.samplers import GammaSampler, AgeAwareBetaSampler

from epiforecast.contacts import ContactGenerator, StaticNetworkTimeSeries, load_edges
from epiforecast.kinetic_model_simulator import KineticModel, print_statuses

from epiforecast.node_identifier_helper import load_node_identifiers
from epiforecast.health_service import HealthService

np.random.seed(12345)
random.seed(1237)

edges = load_edges(os.path.join('..', 'data', 'networks', 'edge_list_SBM_1e3.txt')) 

active_edge_list_frac = 0.034
mean_contact_duration = 1.0 / 1920.0  # unit: days

# construct the generator
contact_generator = ContactGenerator(edges,
                                     active_edge_list_frac,
                                     mean_contact_duration)

# create the list of static networks (for 1 day)
static_intervals_per_day = int(8) #must be int
static_network_interval = 1.0 / static_intervals_per_day
day_of_contact_networks = StaticNetworkTimeSeries(edges)

day_of_contact_networks.add_networks(
        contact_generator.generate_static_networks(averaging_interval = static_network_interval)
        )


# Now get the clinical parameters for the community and health worker populations.
node_identifiers = load_node_identifiers(os.path.join('..', 'data', 'networks', 'node_identifier_SBM_1e3.txt'))

hospital_bed_number = node_identifiers["hospital_beds"].size
health_worker_population = node_identifiers["health_workers"].size
community_population = node_identifiers["community"].size
print("hospital_bed_total " , hospital_bed_number)
population = community_population + health_worker_population

# The age category of each community individual,

age_distribution = [ 0.2352936017,   # 0-19 years
                     0.371604355,    # 20-44 years
                     0.2448085119,   # 45-64 years
                     0.0833381356,   # 65-75 years
                    ]

## 75 onwards
age_distribution.append(1 - sum(age_distribution))

community_ages = populate_ages(community_population, distribution=age_distribution)

#and the ages of the healthcare individual (of working age)
working_age_distribution= [0.0,                 # 0-19 years
                           0.371604355 / (0.371604355 + 0.2448085119),    # 20-44 years
                           0.2448085119 / (0.371604355 + 0.2448085119),  # 45-64 years
                           0.0,                 # 65-75 years
                           0.0]                 # 75+
                            
health_worker_ages = populate_ages(health_worker_population, distribution=working_age_distribution)

ages = np.hstack([health_worker_ages,community_ages])

# Next, we randomly generate clinical properties for our example population.
# Note that the units of 'periods' are days, and the units of 'rates' are 1/day.
latent_periods              = sample_clinical_distribution(GammaSampler(k=1.7, theta=2.0), population=population, minimum=2)
community_infection_periods = sample_clinical_distribution(GammaSampler(k=1.5, theta=2.0), population=population, minimum=1)
hospital_infection_periods  = sample_clinical_distribution(GammaSampler(k=1.5, theta=3.0), population=population, minimum=1)
hospitalization_fraction     = sample_clinical_distribution(AgeAwareBetaSampler(mean=[ 0.02,  0.17,  0.25, 0.35, 0.45], b=4), ages=ages)
community_mortality_fraction = sample_clinical_distribution(AgeAwareBetaSampler(mean=[0.001, 0.001, 0.005, 0.02, 0.05], b=4), ages=ages)
hospital_mortality_fraction  = sample_clinical_distribution(AgeAwareBetaSampler(mean=[0.001, 0.001,  0.01, 0.04,  0.1], b=4), ages=ages)

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

kinetic_model = KineticModel(edges = edges,
                             mean_contact_duration = day_of_contact_networks.contact_networks[0],
                             transition_rates = transition_rates,
                             community_transmission_rate = transmission_rate,
                             hospital_transmission_reduction = hospital_transmission_reduction)

#set up the health service
health_service = HealthService(node_identifiers)


def random_infection_statuses(node_identifiers, initial_infected):
    population = node_identifiers["health_workers"].size + node_identifiers["community"].size
    hospital_bed_number = node_identifiers["hospital_beds"].size 
    number_nodes = hospital_bed_number + population
   
    statuses=np.repeat('S',number_nodes)
    statuses[node_identifiers["hospital_beds"]] = 'P'
    initial_infected_nodes=np.random.choice(population, size=initial_infected, replace=False)
    #Beds can't be infected...
    statuses[hospital_bed_number + initial_infected_nodes] = 'I'

    statuses = { i : statuses[i] for i in np.arange(statuses.size)}
    return statuses

statuses=random_infection_statuses(node_identifiers,100)
print_statuses(statuses)

for i in range(static_intervals_per_day):

    #Based on the current 'node statuses'
    health_service.discharge_and_obtain_patients(statuses)

    kinetic_model.set_mean_contact_duration(day_of_contact_networks.contact_networks[i])
   
    statuses = kinetic_model.simulate(statuses, static_network_interval)

    print_statuses(statuses)
    
