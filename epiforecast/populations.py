from functools import singledispatch

import numpy as np
import networkx as nx

from .samplers import AgeDependentBetaSampler, AgeDependentConstant, BetaSampler, GammaSampler

def sample_distribution(sampler, ages=None, population=None, minimum=0):
    """
    Generate clinical parameters by sampling from a distribution.

    Use cases
    --------

    1. `ages` is not `None`: assume `population = len(ages)`; return an array of size `len(ages)`
       of clinical parameter samples using `minimum + sampler.draw(age)`.

    2. `ages` is None, `population` is not `None`: return an array of size `population` of
       clinical parameter samples using `mininum + sampler.draw()`.

    3. Both `ages` and `population` are `None`: return a single `minimum + sampler.draw()`.

    Args
    ----

    ages (list-like): a list of age categories for the population

    minimum: the minimum value of the statistic (note that this assumes 
             `sampler.draw(age) is always greater than 0.)
             
    sampler: a 'sampler' with a function `sampler.draw(age)` that draws a random
             sample from a distribution, depending on `age`. Samplers that are
             age-independent must still support the syntax `sampler.draw(age)`. 
    """

    if ages is not None:
        return np.array([minimum + sampler.draw(age) for age in ages])
    elif population is not None:
        return np.array([minimum + sampler.draw() for i in range(population)])
    else:
        return minimum + sampler.draw()


# For numpy arrays and constants
@singledispatch
def on_network(parameter, network):
    return parameter

@on_network.register(list)
def list_on_network(parameter, network):
    return np.array(parameter)

@on_network.register(BetaSampler)
@on_network.register(GammaSampler)
def random_sample_on_network(sampler, network):
    return np.array([sampler.minimum + sampler.draw() for node in network.nodes()])

@on_network.register(AgeDependentBetaSampler)
def age_dependent_random_sample_on_network(sampler, network):
    return np.array([sampler.draw(data['age']) for node, data in network.nodes(data=True)])

@on_network.register(AgeDependentConstant)
def age_dependent_on_network(parameter, network):
    return np.array([parameter.constants[data['age']] for node, data in network.nodes(data=True)])

class TransitionRates:
    """
    A container for transition rates.

    Args
    ----
    * population_network (OrderedGraph): Graph whose nodes are people and edges are potential contacts.

    The remaining arguments are either constants, lists, np.array, or samplers from `epiforecast.samplers`:

    * latent_period of infection (1/σ)

    * community_infection_period over which infection persists in the 'community' (1/γ),

    * hospital_infection_period over which infection persists in a hospital setting (1/γ′),

    * hospitalization_fraction, the fraction of infected that become hospitalized (h),

    * community_mortality_fraction, the mortality rate in the community (d),

    * hospital_mortality_fraction, the mortality rate in a hospital setting (d′).
    
    The six transition rates are

    1. Exposed -> Infected
    2. Infected -> Hospitalized
    3. Infected -> Resistant
    4. Hospitalized -> Resistant
    5. Infected -> Deceased
    6. Hospitalized -> Deceased

    These correspond to the dictionaries:

    1. transition_rates.exposed_to_infected
    2. transition_rates.infected_to_hospitalized
    3. transition_rates.infected_to_resistant
    4. transition_rates.hospitalized_to_resistant
    5. transition_rates.infected_to_deceased
    6. transition_rates.hospitalized_to_deceased
    """
    def __init__(self,
                 population_network,
                 latent_periods,
                 community_infection_periods,
                 hospital_infection_periods,
                 hospitalization_fraction,
                 community_mortality_fraction,
                 hospital_mortality_fraction):

        self.population_network = population_network
        self.population = len(population_network)
        self.nodes = nx.convert_node_labels_to_integers(population_network, ordering="sorted")

        # Translate user-input to numpy arrays on the network...
        self.latent_periods               = on_network(latent_periods, population_network)
        self.community_infection_periods  = on_network(community_infection_periods, population_network)
        self.hospital_infection_periods   = on_network(hospital_infection_periods, population_network)
        self.hospitalization_fraction     = on_network(hospitalization_fraction, population_network)
        self.community_mortality_fraction = on_network(community_mortality_fraction, population_network)
        self.hospital_mortality_fraction  = on_network(hospital_mortality_fraction, population_network)

        self._calculate_transition_rates()

    def _calculate_transition_rates(self):
        """
        Calculates the transition rates, given the current clinical parameters.
        If the clinical parameter is only a single value, we apply it to all nodes.
        """
        σ = 1 / self.latent_periods
        γ = 1 / self.community_infection_periods
        h = self.hospitalization_fraction
        d = self.community_mortality_fraction
        γ_prime = 1 / self.hospital_infection_periods
        d_prime = self.hospital_mortality_fraction

        # Broadcast to arrays of size `population`
        σ *= np.ones(self.population)
        γ *= np.ones(self.population)
        h *= np.ones(self.population)
        d *= np.ones(self.population)
        γ_prime *= np.ones(self.population)
        d_prime *= np.ones(self.population)

        self.exposed_to_infected       = { node: σ[i]                          for i, node in enumerate(self.nodes) }
        self.infected_to_resistant     = { node: (1 - h[i] - d[i]) * γ[i]      for i, node in enumerate(self.nodes) }
        self.infected_to_hospitalized  = { node: h[i] * γ[i]                   for i, node in enumerate(self.nodes) }
        self.infected_to_deceased      = { node: d[i] * γ[i]                   for i, node in enumerate(self.nodes) }

        self.hospitalized_to_resistant = { node: (1 - d_prime[i]) * γ_prime[i] for i, node in enumerate(self.nodes) }
        self.hospitalized_to_deceased  = { node: d_prime[i] * γ_prime[i]       for i, node in enumerate(self.nodes) }

    def set_clinical_parameter(self, parameter, values):
        setattr(self, parameter, values)
        self._calculate_transition_rates()




def populate_ages(population, distribution):
    """
    Returns a numpy array of length `population`, with age categories from
    0 to len(`distribution`), where the elements of `distribution` specify
    the probability that an individual's age falls into the corresponding category.

    Args
    ----
    population (int): Number of people in the network

    population (int): Number of people in the network
    """
    classes = len(distribution)

    ages = [np.random.choice(np.arange(classes), p=distribution) 
            for person in range(population)]

    return np.array(ages)




def assign_ages(population_network, distribution, distribution_HCW, node_identifiers):
    """
    Assigns ages to the nodes of `population_network` according to `distribution`.

    Args
    ----
    population_network (networkx Graph): A graph representing a community and its contacts.

    distribution (list-like): A list of quantiles (for community). Must sum to 1.
    
    distribution_HCW (list-like): A list of quantiles (for health workers). Must sum to 1.
    
    node_identifiers (dictionary): A dictionary that contains node identifiers of
                                   health worker and community nodes.

    Example
    -------

    distribution = [0.25, 0.5, 0.25] # a simple age distribution
    distribution_HCW = [0, 1, 0] # a simple age distribution for health workers
    population_network = nx.barabasi_albert_graph(100, 2)
    node_identifiers = load_node_identifiers(...)
    assign_ages(population_network, distribution, distribution_HCW, node_identifiers)
    """    
    health_workers = node_identifiers['health_workers'].size
    community = node_identifiers['community'].size

    ages_HCW = populate_ages(health_workers, distribution=distribution_HCW)
    ages_community = populate_ages(community, distribution=distribution)
    
    nodes_HCW = range(health_workers)
    nodes_community = range(health_workers, community+health_workers)

    nodal_ages = { node: ages_HCW[i] for i, node in enumerate(nodes_HCW) }
    nodal_ages.update( {node: ages_community[i] for i, node in enumerate(nodes_community)} )

    nx.set_node_attributes(population_network, values=nodal_ages, name='age')
