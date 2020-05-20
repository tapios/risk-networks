# risk-networks

Code for risk networks: a blend of compartmental models, graphs, data assimilation and semi-supervised learning.

- Dependencies:
  - cycler==0.10.0
  - eon==1.1
  - kiwisolver==1.2.0
  - matplotlib==3.2.1
  - networkx==2.4
  - numpy==1.18.3
  - pyparsing==2.4.7
  - pytz==2019.3
  - scipy==1.4.1


- Added conda environment `yml` to have the bare minimum of `python` modules to
  work. To replicate the environment make sure you have anaconda preinstalled
  and use the following command from within the repo directory (or specify the
  full path to the yml file):
  <!--  -->
  ```{bash}
  conda env create -f risknet.yml
  ```

# Overview

This overview

1. Performs an simulation of an example epidemic

    1. Defines a population, including:
        * Age distribution,
        * Age-depednent clinical statistics,
        * Transmission rate.

    2. Generates a time-averaged contact network.

    3. Simulates an epidemic with a stochastic kinetic model.

    4. Simulates an epidemic with a deterministic 'risk', or 'master equation' model.

2. Performs data assimilation over a one-day window for the same population, but a different scenario.

## Import package functionality

We first import the package's functionality:

```python
# Utilities for generating random populations
from epiforecast.populations import populate_ages, ClinicalStatistics, TransitionRates
from epiforecast.samplers import GammaSampler, AgeAwareBetaSampler

# Function that generates a time-averaged contact network from a rapidly-fluctuating
# birth-death process.
from epiforecast.contacts import generate_time_averaged_contact_network

# Kinetic model simulation tool
from epiforecast.kinetic_simulation import KineticModel

# Simulation tool for ensembles of master equation models
from epiforecast.risk_simulation import MasterEquationModelEnsemble

# Tools for data assimilation and performing observations of 'synthetic data'
# generated by the KineticModel
from epiforecast.data_assimilation import DataAssimilator, EnsembleAdjustedKalmanFilter
from epiforecast.data_assimilation import SixHourlyTotalStateObservations

# Tools for simulating specific scenarios
from epiforecast.scenarios import random_infection
from epiforecast.scenarios import midnight_on_Tuesday
from epiforecast.scenarios import state_distribution_at_midnight_on_Tuesday
from epiforecast.scenarios import transition_rates_distribution_at_midnight_on_Tuesday
from epiforecast.scenarios import transmission_rates_distribution_at_midnight_on_Tuesday
```

## Example simulation of an epidemic

An epidemic unfolds on a time-evolving contact network, in a population
with a distribution of clinical and transmission properties.

### Define the 'population' and its clinical characteristics

First we define the population by the number of individuals:

```python
population = 1000
```

and the age category of each individual,

```python
age_distribution = [ 0.23,  # 0-19 years
                     0.39,  # 20-44 years
                     0.25,  # 45-64 years
                     0.079  # 65-75 years
                    ]  

# 75 onwards                    
age_distribution.append(1 - sum(age_distribution))

ages = populate_ages(population, distribution=age_distribution)
```

In the above we define the 6 age categories 0-19, 20-44, 45-64, 65-74, 75->.

Next we define six 'clinical statistics'.
Clinical statistics are individual properties that determine
their recovery rate and risk of becoming hospitalized or dying, for example.
The six clinical statistics are

1. `latent_period` of infection (`σ⁻¹`)
2. `community_infection_period` over which infection persists in the 'community' (`γ`),
3. `hospital_infection_period` over which infection persists in a hospital setting (`γ′`),
4. `hospitalization_fraction`, the fraction of infected that become hospitalized (`h`),
5. `community_mortality_fraction`, the mortality rate in the community (`d`),
6. `hospital_mortality_fraction`, the mortality rate in a hospital setting (`d′`).

We randomly generate clinical properties for our example population,

```python
latent_periods              = ClinicalStatistic(ages = ages, const = 2, 
                                                sampler = GammaSampler(k=1.7, theta=2.0))

community_infection_periods = ClinicalStatistic(ages = ages, const = 1,
                                                sampler = GammaSampler(k=1.5, theta=2.0))

hospital_infection_periods  = ClinicalStatistic(ages = ages, const = 1, 
                                                sampler = GammaSampler(k=1.5, theta=3.0))

hospitalization_fraction = ClinicalStatistic(ages = ages,
    sampler = AgeAwareBetaSampler(mean=[ 0.02,  0.17,  0.25, 0.35, 0.45], b=4))

community_mortality_fraction = ClinicalStatistic(ages = ages,
    sampler = AgeAwareBetaSampler(mean=[0.001, 0.001, 0.005, 0.02, 0.05], b=4))

hospital_mortality_fraction  = ClinicalStatistic(ages = ages,
    sampler = AgeAwareBetaSampler(mean=[0.001, 0.001,  0.01, 0.04,  0.1], b=4))
```

`AgeAwareBetaSampler` is a generic sampler of statistical distribution with a function `beta_sampler.draw(age)`
which generates clinical properties for each individual based on their `age` class (see `numpy.random.beta`
for more information). `const` is a random number to which `gamma_sampler.draw()` or `beta_sampler.draw(age)`
is added.

We process the clinical data to determine transition rates between each
epidemiological state,

```python
transition_rates = TransitionRates(latent_periods,
                                   community_infection_periods,
                                   hospital_infection_periods,
                                   fractional_hospitalization_rates,
                                   community_mortality_rates,
                                   hospital_mortality_rates)
```

The transition rates have units `1 / day`. There are six transition rates:

1. Exposed -> Infected
2. Infected -> Hospitalized
3. Infected -> Resistant
4. Hospitalized -> Resistant
5. Infected -> Deceased
6. Hospitalized -> Deceased

### Define the transmission rates

In general, the transmission rate is different for each _pair_ of individuals, and is
therefore can be as large as `population * (population - 1)`.
The transmission rate may depend on properties specific to each individual in each pair,
such as the amount of protective equipment each individual wears.
Here, we pick an arbitrary constant,

```python
constant_transmission_rate = 0.1 # per average number of contacts per day
```

The `transition_rates` and `constant_transmission_rate` define the epidemiological 
characteristics of the population.

### Generation of a time-evolving contact network

Physical contact between people in realistic communities is rapidly evolving.
We average the contact time between individuals over a `static_contacts_interval`,
over which, for the purposes of solving both the kinetic and master equations,
we assume that the graph of contacts is static:

```python
day = 1.0 # We use time units of "day"s
static_contacts_interval = day / 4
```

On a graph, or 'network', individuals are nodes and contact times are the weighted edges
between them. We create a contact network averaged over `static_contacts_interval`
for a population of 1000 with diurnally-varying mean contact rate,

```python
diurnal_contacts_modulation = lambda t, λᵐⁱⁿ, λᵐᵃˣ: np.max([λᵐⁱⁿ, λᵐᵃˣ * (1 - np.cos(np.pi * t)**2)])

network_generator = EvolvingContactNetworkGenerator(
                                         population = population,  
                                  start_time_of_day = 0.5, # half-way through the day, aka 'high noon'
                                 averaging_interval = static_contacts_interval,
                                   transition_rates = transition_rates,
                                         lambda_min = 5,
                                         lambda_max = 22,
                   initial_fraction_of_active_edges = 0.034,
                               measurement_interval = 0.1,
                              mean_contact_duration = 10 / 60 / 24, # 10 minutes
                        diurnal_contacts_modulation = diurnal_contacts_modulation,
                        # **other_contact_network_generation_parameters? 
)

contact_network = network_generator.generate_time_averaged_contact_network( **network_generation_parameters   )
```

We have included the abstract `network_generation_parameters` as a placeholder for
a dictionary of parameters that can characterize the contact network.

### Simulation of an epidemic

With a population, its clinical properties, a transmission rate, and a contact network,
we can now simulate an epidemic over the `static_contacts_interval`.

We seed the epidemic by randomly infecting a small number of individuals,

```python
initial_state = random_infection(population, infected=20)
```

#### Kinetic simulation

A kinetic simulation is direct simulation of the stochastic evolution of an epidemic.

```python
kinetic_model = KineticModel(       contact_network = contact_network,
                                 transmission_rates = constant_transmission_rate,
                             state_transition_rates = transition_rates
                            )

# Set the current state of the kinetic model
kinetic_model.set_state(initial_state)

# Simulate an epidemic over the static_contacts_interval
output = kinetic_model.simulate(static_contacts_interval)
```

#### Simulation using the mean-field master equations

The mean field equations represent the average behavior of many stochastic epidemics.
Alternatively, we can interpret the mean-field state as the 'probability' that each individual
has a certain epidemiological state.

For data assimilation, we simulate an *ensemble* of master equation models.
For this example, we conduct a forward run of a single master equation model.

```python
master_model = MasterEquationModelEnsemble(          ensemble_size = 1,
                                                   contact_network = contact_network,
                                                transmission_rates = constant_transmission_rate,
                                            state_transition_rates = transition_rates,
                                            # can also define mean_field_closure here
                                           )

# Set the master equation state to the same initial state as the kinetic model.
master_model.set_state(initial_state)

# Simulate an epidemic over the static_contacts_interval.
output = master_model.simulate(static_contacts_interval)
```

(We can then make a plot that compares the evolution of the epidemic in the two models,
and paste it here.)

## Data assimilation

Here we demonstrate forward filter data assimilation over a single assimilation 'window',
on a specific epidemic scenario. Our experiment begins at midnight (hour 00) on Tuesday,
and proceeds for 1 day until hour 24 (midnight on Wednesday). We assimilate observations
every 6 hours. Recall that `static_contacts_interval = day / 4`,

```python
static_contacts_interval = day / 4
data_assimilation_window = 1 * day
intervals_per_window = int(data_assimilation_window / static_contacts_interval)
```

We begin the experiment by simulating the slow evolution of a contact network over one day:

```python
network_generator = EvolvingContactNetworkGenerator(
                                                 population = population,
                                          start_time_of_day = 0,
                                         averaging_interval = static_contacts_interval,
                                           transition_rates = transition_rates,
                                                 lambda_min = 5,  # contacts per day during activity minimum
                                                 lambda_max = 22, # contacts per day during activity maximum
                           initial_fraction_of_active_edges = 0.034,
                                       measurement_interval = 0.1,
                                      mean_contact_duration = 10 / 60 / 24, # 10 minutes
                                diurnal_contacts_modulation = diurnal_contacts_modulation,
                        # **other_contact_network_generation_parameters?
)
    
static_contacts_times = []
contact_networks = []


# Generate 4 contact networks for hours 00-06, 06-12, 12-18, 18-24
for i in range(intervals_per_window):
    contact_network = network_generator.generate_time_averaged_contact_network(start_time_of_day=i*static_contacts_interval)
    contact_networks.append(contact_network)                                  
    static_contacts_times.append(i * static_contacts_interval)

# Initialize the kinetic model, using the initial contact network at hour 00    
kinetic_model = KineticModel(       contact_network = contact_networks[0],
                                 transmission_rates = constant_transmission_rate,
                             state_transition_rates = transition_rates
                            )

# Instantiate an example epidemic state
state = midnight_on_Tuesday(kinetic_model)

# Set the current state of the kinetic model
kinetic_model.set_state(state)

synthetic_data = []

for i in range(intervals_per_window):
    # Set the contact network for the kinetic model
    kinetic_model.set_contact_network(contact_networks[i])

    # Run the kinetic model forward for 6 hours
    kinetic_model.simulate(static_contacts_interval)

    # Record the output
    synthetic_data.append(kinetic_model.state())
```

We now have 'data' from hours 06, 12, 18, and 24.

```python
master_model = MasterEquationModelEnsemble(        contact_network = contact_networks[0],
                                                     ensemble_size = 20,
                                                transmission_rates = constant_transmission_rate,
                                            state_transition_rates = transition_rates,
                                           )

# Generate a joint distribution of states and transition rates for this example.
state_distribution              = state_distribution_at_midnight_on_Tuesday()
transition_rates_distribution   = transition_rates_distribution_at_midnight_on_Tuesday()
transmission_rates_distribution = transmission_rates_distribution_at_midnight_on_Tuesday()

master_model.populate_states(distribution=state_distribution)
master_model.populate_transition_rates(distribution=transition_rates_distribution)
master_model.populate_transmission_rates(distribution=transmission_rates_distribution)

# Initialize the data assimilation method
assimilator = DataAssimilator(   observations = SixHourlyTotalStateObservations(),
                                       method = EnsembleAdjustedKalmanFilter(),
                             )

for i in range(intervals_per_window):
    # Set the contact network for the ensemble of master equation models
    master_model.set_contact_network(contact_networks[i])

    # Run the master model ensemble forward for six hours
    master_model.simulate(static_contacts_interval)  

    new_transition_rates, new_transmission_rates, new_ensemble = assimilator.update(network_generator.get_contact_networks()
                                                                                    master_model.ensemble,        
                                                                                    synthetic_data)

    # Update the master model ensemble and parameters
    master_model.set_ensemble(new_ensemble)
    master_model.set_transition_rates(new_transition_rates)
    master_model.set_transmission_rates(new_transmission_rates)
```

This completes data assimilation over one 'assimilaton window'. To apply interventions, we adjust
the inputs to `network_generator.generate_time_averaged_contact_network`, generate a new time series of
contact networks, and then simulate the evolution of the epidemic over a subsequent assimilation window.

(Now we should compare the ensemble trajectory and the trajectory of the synthetic data...)
