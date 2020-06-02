import os, sys; sys.path.append(os.path.join(".."))

# ## Import package functionality
# 
# We first import the package's functionality:

# Utilities for generating random populations
from epiforecast.populations import populate_ages, sample_distribution, TransitionRates
from epiforecast.samplers import GammaSampler, AgeDependentBetaSampler

# Function that generates a time-averaged contact network from a rapidly-fluctuating
# birth-death process.
# What to import here?

# Kinetic model simulation tool
from epiforecast.kinetic_model_simulator import KineticModel

# Simulation tool for ensembles of master equation models
from epiforecast.risk_simulator import MasterEquationModelEnsemble

# Tools for data assimilation and performing observations of 'synthetic data'
# generated by the KineticModel
from epiforecast.data_assimilator import DataAssimilator
from epiforecast.observations import FullObservation

# Tools for simulating specific scenarios
from epiforecast.scenarios import random_infection, randomly_infected_ensemble
from epiforecast.scenarios import percent_infected_at_midnight_on_Tuesday
from epiforecast.scenarios import ensemble_transition_rates_at_midnight_on_Tuesday
from epiforecast.scenarios import ensemble_transmission_rates_at_midnight_on_Tuesday

from epiforecast.node_identifier_helper import load_node_identifiers

# ## Example simulation of an epidemic


np.random.seed(12345)
random.seed(1237)
# An epidemic unfolds on a time-evolving contact network, in a population
# with a distribution of clinical and transmission properties.
# 
# ### Define the 'population' and its clinical characteristics
# The total number of nodes are hospital beds, health workers and community types
# First we load the population by the number of different individuals:
node_identifiers = load_node_identifiers(os.path.join('..', 'data', 'networks', 'node_identifier_SBM_1e3.txt'))

hospital_bed_number = node_identifiers["hospital_beds"].size
health_worker_population = node_identifiers["health_workers"].size
community_population = node_identifiers["community"].size

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
# In the above we define the 6 age categories 0-19, 20-44, 45-64, 65-74, 75->.

# Next we define six 'clinical statistics'.
# Clinical statistics are individual properties that determine
# their recovery rate and risk of becoming hospitalized or dying, for example.
# The six clinical statistics are
# 
# 1. `latent_period` of infection (`1/σ`)
# 2. `community_infection_period` over which infection persists in the 'community' (`1/γ`),
# 3. `hospital_infection_period` over which infection persists in a hospital setting (`1/γ′`),
# 4. `hospitalization_fraction`, the fraction of infected that become hospitalized (`h`),
# 5. `community_mortality_fraction`, the mortality rate in the community (`d`),
# 6. `hospital_mortality_fraction`, the mortality rate in a hospital setting (`d′`).

# We randomly generate clinical properties for our example population,

latent_periods              = sample_distribution(GammaSampler(k=1.7, theta=2.0), population=population, minimum=2)
community_infection_periods = sample_distribution(GammaSampler(k=1.5, theta=2.0), population=population, minimum=1)
hospital_infection_periods  = sample_distribution(GammaSampler(k=1.5, theta=3.0), population=population, minimum=1)

hospitalization_fraction     = sample_distribution(AgeDependentBetaSampler(mean=[ 0.02,  0.17,  0.25, 0.35, 0.45], b=4), ages=ages)
community_mortality_fraction = sample_distribution(AgeDependentBetaSampler(mean=[0.001, 0.001, 0.005, 0.02, 0.05], b=4), ages=ages)
hospital_mortality_fraction  = sample_distribution(AgeDependentBetaSampler(mean=[0.001, 0.001,  0.01, 0.04,  0.1], b=4), ages=ages)

# `AgeDependentBetaSampler` is a generic sampler of statistical distribution with a function `beta_sampler.draw(age)`
# which generates clinical properties for each individual based on their `age` class (see `numpy.random.beta`
# for more information). `minimum` is a random number to which `gamma_sampler.draw()` or `beta_sampler.draw(age)`
# is added.

# We process the clinical data to determine transition rates between each
# epidemiological state,

transition_rates = TransitionRates(population,
                                   latent_periods,
                                   community_infection_periods,
                                   hospital_infection_periods,
                                   hospitalization_fraction,
                                   community_mortality_fraction,
                                   hospital_mortality_fraction)

# The transition rates have units `1 / day`. There are six transition rates:
# 
# 1. Exposed -> Infected
# 2. Infected -> Hospitalized
# 3. Infected -> Resistant
# 4. Hospitalized -> Resistant
# 5. Infected -> Deceased
# 6. Hospitalized -> Deceased

# ### Define the transmission rates

# In general, the transmission rate is different for each _pair_ of individuals, and is
# jtherefore can be as large as `population * (population - 1)`.
# The transmission rate may depend on properties specific to each individual in each pair,
# uch as the amount of protective equipment each individual wears.
# ere, we pick an arbitrary constant,

constant_transmission_rate = 0.1 # per average number of contacts per day

# The `transition_rates` and `constant_transmission_rate` define the epidemiological
# characteristics of the population.
# 
# ### Generation of a time-evolving contact network
# 
# Physical contact between people in realistic communities is rapidly evolving.
# We average the contact time between individuals over a `static_contacts_interval`,
# over which, for the purposes of solving both the kinetic and master equations,
# we assume that the graph of contacts is static:

day = 1.0 # We use time units of "day"s
static_contacts_interval = day / 4

# On a graph, or 'network', individuals are nodes and contact times are the weighted edges
# between them. We create a contact network averaged over `static_contacts_interval`
# for a population of 1000 with diurnally-varying mean contact rate,

diurnal_contacts_modulation = lambda t, λᵐⁱⁿ, λᵐᵃˣ: np.max([λᵐⁱⁿ, λᵐᵃˣ * (1 - np.cos(np.pi * t)**2)])

#network_generator = EvolvingContactNetworkGenerator(
#                                         population = population,
#                                  start_time_of_day = 0.5, # half-way through the day, aka 'high noon'
#                                 averaging_interval = static_contacts_interval,
#                                   transition_rates = transition_rates,
#                                         lambda_min = 5,
#                                         lambda_max = 22,
#                   initial_fraction_of_active_edges = 0.034,
#                               measurement_interval = 0.1,
#                              mean_contact_duration = 10 / 60 / 24, # 10 minutes
#                        diurnal_contacts_modulation = diurnal_contacts_modulation,
#)
#
#contact_network = network_generator.generate_time_averaged_contact_network( **network_generation_parameters   )
