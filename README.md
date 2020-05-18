# risk-networks

Code for risk networks: a blend of compartmental models, graphs, data assimilation and semi-supervised learning

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

This code does four things:

1. Generation of an epidemic scenario
2. Kinetic model simulation
3. Master equation forward and backward simulation
4. Data assimilation

## Generation of epidemic scenario

An epidemic unfolds on a time-evolving contact network, in a population
with a distribution of clinical and transmission properties.

### Define the 'population' and its clinical characteristics

First we define the population by the number of individuals. Here we use a population of 1000.

```python
population = 1000
```

*We may need to define the ages or age category of each node first, eg*

```python
ages = label_distribution(population, labels=6)
```

Next we define the six clinical properties of the population:

1. `latent_period` of infection,
2. `community_infection_period` over which infection persists in the 'community',
3. `hospital_infection_period` over which infection persists in a hospital setting,
4. `fractional_hospitalization_rate`, the rate at which infected people become hospitalized,
5. `community_mortality_rate`, the mortality rate in the community,
6. `hospital_mortality_rate`, the mortality rate in a hospital setting.

The clinical properties of our population are generated random from statistical distributions:

```python
latent_periods                   = VariableClinicalCharacteristic(ages = ages, sampler = AgeAwareSampler(*sampler_properties))
community_infection_periods      = VariableClinicalCharacteristic(ages = ages, sampler = AgeAwareSampler(*sampler_properties))
hospital_infection_periods       = VariableClinicalCharacteristic(ages = ages, sampler = AgeAwareSampler(*sampler_properties))
fractional_hospitalization_rates = VariableClinicalCharacteristic(ages = ages, sampler = AgeAwareSampler(*sampler_properties))
community_mortality_rates        = VariableClinicalCharacteristic(ages = ages, sampler = AgeAwareSampler(*sampler_properties))
hospital_mortality_rates         = VariableClinicalCharacteristic(ages = ages, sampler = AgeAwareSampler(*sampler_properties))
```

The `AgeAwareSampler` is a generic sampler of statistical distribution with a function `sampler.draw(age)`
which generates clinical properties for each individual based on their `age` class.

We collect all of clinical data into one object that defines the transition rates between each
epidemiological state,

```python
transition_rates = TransitionRates(latent_periods,
                                   community_infection_periods,
                                   hospital_infection_periods,
                                   fractional_hospitalization_rates,
                                   community_mortality_rates,
                                   hospital_mortality_rates)
```

### Define the transmission rates

In general, the transmission rate can have a different value for each _pair_ of individuals, and is
therefore can be as large as `population**2`. Another possibility is that the transmission rate is
a function of individual properties (such as the amount of protective equipment an individual wears)
and that the transmission rate is a function of these individual-level properties. Here, we assume 
that it is constant for each pair of individuals. We choose the 'effective transmission rate' to 
reproduce a realistic epidemic in LA county.

```python
constant_transmission_rate = 0.1
```

The `transition_rates` and `constant_transmission_rate` define the clinical characteristics of the population.

### Generation of a time-evolving contact network

The network of human contacts is rapidly evolving in a realistic population.
We average the contact rate of each individual over a `static_contacts_interval`,
over which, for the purposes of solving the master equations, we assume that
the graph of contacts is static.

We therefore first specify the `static_contacts_interval`:

```python
day = 1.0 # We use time units of "day"s
static_contacts_interval = day / 4
```

For example, to create a contact network averaged over `static_contacts_interval`
for a population of 1000 people we write

```python
contact_network = generate_time_averaged_contact_network(
                                population = population,
                        averaging_interval = static_contacts_interval,
                                             **contact_network_generation_parameters,
)
```

Note that `contact_network_generation_parameters` is a dictionary of parameters that
characterize the statistics of the (randomly generated) contact network.

### Simulation of an epidemic

We can now simulate an epidemic over the `static_contacts_interval`.

For that, we generate an initial state corresponding to an epidemic seeded by
a small number of randomly infected individuals.

```python
initial_state = random_infection(population, infected=10)
```

#### Kinetic simulation

A kinetic simulation is direct simulation of the stochastic evolution of an epidemic.

```python
kinetic_model = KineticEquationModel(contact_network,
                                     transmission_rates = constant_transmission_rate,
                                     state_transition_rates = transition_rates)

# Set the current state of the kinetic model
kinetic_model.set_state(initial_state)

# Simulate an epidemic over the static_contacts_interval
output = kinetic_model.simulate(static_contacts_interval)
```

#### Simulation using the mean-field master equations

For data assimilation, we need to simulate an *ensemble* of master equation
models. Here we conduct a forward run of a single master equation model.

```python
master_model = MasterEquationModelEnsemble(contact_network,
                                                     ensemble_size = 1,
                                                transmission_rates = constant_transmission_rate,
                                            state_transition_rates = transition_rates,
                                            # can also define mean_field_closure here
                                           )

# Set the current state of the master equation ensemble
master_model.set_state(initial_state)

# Simulate an epidemic over the static_contacts_interval
output = master_model.simulate(static_contacts_interval)
```

(We can then make a plot that compares the evolution of the epidemic in the two models.)

## Data assimilation

For data assimilation, we run an ensemble of master equation models alongside a kinetic model
that generates 'observations'.

The primary parameter for data assimilation is the window over which we collect observations.

```python
data_assimilation_window = 1 * day
```

Next we initialize the observations.

```python
# Reset the state of the kinetic model to the initial state
kinetic_model.set_state(initial_state)

observations = Observations(kinetic_model, **observation_parameters)
```

(This is a work in progress...)
