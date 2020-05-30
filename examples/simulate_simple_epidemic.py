import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer

import networkx as nx
import numpy as np
import random 
import matplotlib.pyplot as plt

# Utilities for generating random populations
from epiforecast.populations import populate_ages, sample_distribution, TransitionRates
from epiforecast.samplers import GammaSampler, AgeAwareBetaSampler

from epiforecast.contact_simulator import ContactSimulator, DiurnalMeanContactRate
from epiforecast.kinetic_model_simulator import KineticModel, print_statuses
from epiforecast.scenarios import load_edges

from epiforecast.node_identifier_helper import load_node_identifiers

def random_epidemic(population, initial_infected, initial_exposed):
    """
    Returns a status dictionary associated with a random infection
    within a population associated with node_identifiers.
    """

    statuses = {node: 'S' for node in range(population)}

    initial_infected_nodes = np.random.choice(population, size=initial_infected, replace=False)
    initial_exposed_nodes = np.random.choice(population, size=initial_exposed, replace=False)

    for i in initial_infected_nodes:
        statuses[i] = 'I'

    for i in initial_exposed_nodes:
        if statuses[i] is not 'I':
            statuses[i] = 'E'

    return statuses

#
# Set random seeds for reproducibility
#

# Both numpy.random and random are used by the KineticModel.
np.random.seed(123)
random.seed(123)

#
# Load an example network
#

edges = load_edges(os.path.join('..', 'data', 'networks', 'edge_list_SBM_1e3.txt')) 

contact_network = nx.Graph()
contact_network.add_edges_from(edges)
population = len(contact_network)

#
# Build the contact simulator
#

contact_simulator = ContactSimulator(

             n_contacts = nx.number_of_edges(contact_network),
    mean_event_duration = 1 / 60 / 24, # 1 minute in units of days
      mean_contact_rate = DiurnalMeanContactRate(maximum=40, minimum=2),
             start_time = -3 / 24, # negative start time allows short 'spinup' of contacts

)

#
# Clinical parameters of an age-distributed population
#

# The age category of each community individual,
age_distribution = [ 
                    0.24,   # 0-19 years
                    0.37,   # 20-44 years
                    0.24,   # 45-64 years
                    0.083,  # 65-75 years
                   ]

## 75 onwards
age_distribution.append(1 - sum(age_distribution))
                       
ages = populate_ages(population, distribution=age_distribution)

# Next, we randomly generate clinical properties for our example population.
# Note that the units of 'periods' are days, and the units of 'rates' are 1/day.
latent_periods               = sample_distribution(GammaSampler(k=1.7, theta=2.0), minimum=2, population=population)
community_infection_periods  = sample_distribution(GammaSampler(k=1.5, theta=2.0), minimum=1, population=population)
hospital_infection_periods   = sample_distribution(GammaSampler(k=1.5, theta=3.0), minimum=1, population=population)
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
# Build the kinetic model
#

kinetic_model = KineticModel(contact_network = contact_network,
                             transition_rates = transition_rates,
                             community_transmission_rate = transmission_rate,
                             hospital_transmission_reduction = hospital_transmission_reduction)

# 
# Seed an infection
#

statuses = random_epidemic(population, initial_infected=20, initial_exposed=100)
kinetic_model.set_statuses(statuses)

# 
# Simulate the uncontrolled growth of an epidemic for 7 days
#

static_contact_interval = 3/24 # days
growth_interval = 7 # days

growth_steps = int(growth_interval / static_contact_interval)
growth_start_times = static_contact_interval * np.arange(growth_steps)

# Run the simulation
for i in range(growth_steps):


    start = timer() 
    contacts = contact_simulator.mean_contact_duration(stop_time = growth_start_times[i])
    end = timer() 

    print("Simulating an epidemic",
          "from day {:.3f}".format(growth_start_times[i]),
          "until day {:.3f}.".format(growth_start_times[i] + static_contact_interval), 
          "Contact simulation took {:.3f} seconds".format(end - start))

    kinetic_model.set_mean_contact_duration(contacts)
    kinetic_model.simulate(static_contact_interval)

#
# Plot the results.
#

fig, axs = plt.subplots(nrows=2, sharex=True)

plt.sca(axs[0])
plt.plot(kinetic_model.times, kinetic_model.statuses['S'])
plt.ylabel("Number of Susceptible (S)")

plt.sca(axs[1])
plt.plot(kinetic_model.times, kinetic_model.statuses['E'], label='Exposed')
plt.plot(kinetic_model.times, kinetic_model.statuses['I'], label='Infected')
plt.plot(kinetic_model.times, kinetic_model.statuses['H'], label='Hospitalized')
plt.plot(kinetic_model.times, kinetic_model.statuses['R'], label='Resistant')
plt.plot(kinetic_model.times, kinetic_model.statuses['D'], label='Deceased')

plt.xlabel("Time (days)")
plt.ylabel("Number of E, I, H, R, D")
plt.legend()

plt.show()
