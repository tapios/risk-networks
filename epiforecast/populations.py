from functools import singledispatchmethod

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
    def __init__(
            self,
            population,
            lp_sampler,
            cip_sampler,
            hip_sampler,
            hf_sampler,
            cmf_sampler,
            hmf_sampler):
        """
        Constructor with samplers

        For readability, long names are abbreviated using the following
        glossary:
            lp  :   latent period
            cip :   community infection period
            hip :   hospital infection period
            hf  :   hospitalization fraction
            cmf :   community mortality fraction
            hmf :   hospital mortality fraction

        Input:
            population (int): population count
            *_sampler (int),
                      (float)   : a constant value for a parameter
                      (list)    : a list of parameters of length population
                      (np.array): (population,) array of parameters
                      (BetaSampler),
                      (GammaSampler),
                      (AgeDependentConstant),
                      (AgeDependentBetaSampler): a sampler to use
        """
        # TODO extract clinical parameters into its own class
        self.population  = population
        self.lp_sampler  = lp_sampler
        self.cip_sampler = cip_sampler
        self.hip_sampler = hip_sampler
        self.hf_sampler  = hf_sampler
        self.cmf_sampler = cmf_sampler
        self.hmf_sampler = hmf_sampler

        self.latent_periods               = None
        self.community_infection_periods  = None
        self.hospital_infection_periods   = None
        self.hospitalization_fraction     = None
        self.community_mortality_fraction = None
        self.hospital_mortality_fraction  = None

        self.exposed_to_infected       = None
        self.infected_to_resistant     = None
        self.infected_to_hospitalized  = None
        self.infected_to_deceased      = None
        self.hospitalized_to_resistant = None
        self.hospitalized_to_deceased  = None

    @singledispatchmethod
    @staticmethod
    def __draw_using(parameter, distr_parameters):
        raise ValueError(
                self.__class__.__name__
                + ": this type of argument is not supported: "
                + parameter.__class__.__name__)

    @__draw_using.register(int)
    @__draw_using.register(float)
    @staticmethod
    def __draw_using_const(parameter, distr_parameters):
        return parameter

    @__draw_using.register(list)
    @staticmethod
    def __draw_using_list(parameter_list, distr_parameters):
        return np.array(parameter_list)

    @__draw_using.register(np.ndarray)
    @staticmethod
    def __draw_using_array(parameter_array, distr_parameters):
        return parameter_array

    @__draw_using.register(BetaSampler)
    @__draw_using.register(GammaSampler)
    @staticmethod
    def __draw_using_sampler(sampler, distr_parameters):
        return np.array([sampler.minimum+sampler.draw() for dp in distr_parameters])

    @__draw_using.register(AgeDependentBetaSampler)
    @staticmethod
    def __draw_using_age_beta_sampler(sampler, distr_parameters):
        return np.array([sampler.draw(dp) for dp in distr_parameters])

    @__draw_using.register(AgeDependentConstant)
    @staticmethod
    def __draw_using_age_const_sampler(sampler, distr_parameters):
        return np.array([sampler.constants[dp] for dp in distr_parameters])

    def draw_and_set_clinical_using(
            self,
            distr_parameters):
        """
        Draw and set clinical parameters using distributional parameters

        Samplers provided in the constructor for clinical parameter (periods and
        fractions) might be either distributions that depend on certain
        parameters (say, mean and variance in the Gaussian case), or arrays, or
        constant values.
        This method draws from those distributions according to the provided
        distributional parameters.
        For arrays, they are left unchanged.
        For constant values, they are propagated to arrays.
        In either case, the resulting clinical parameters will be of size
            distr_parameters.size

        Note that this method is NOT idempotent, i.e. in the following
            transition_rates.draw_and_set_clinical_using(dp)
            transition_rates.draw_and_set_clinical_using(dp)
        second call draws new samples (for those clinical parameters which had
        samplers specified in the constructor).
        It leaves others unchanged though.

        Input:
            distr_parameters (np.array): (n,) array of parameters

        Output:
            None
        """
        self.latent_periods               = self.__draw_using(
                self.lp_sampler,
                distr_parameters)
        self.community_infection_periods  = self.__draw_using(
                self.cip_sampler,
                distr_parameters)
        self.hospital_infection_periods   = self.__draw_using(
                self.hip_sampler,
                distr_parameters)
        self.hospitalization_fraction     = self.__draw_using(
                self.hf_sampler,
                distr_parameters)
        self.community_mortality_fraction = self.__draw_using(
                self.cmf_sampler,
                distr_parameters)
        self.hospital_mortality_fraction  = self.__draw_using(
                self.hmf_sampler,
                distr_parameters)

    def calculate_from_clinical(self):
        """
        Calculate transition rates using the current clinical parameters

        Output:
            None
        """
        σ       = self.__convert_to_array(1 / self.latent_periods)
        γ       = self.__convert_to_array(1 / self.community_infection_periods)
        γ_prime = self.__convert_to_array(1 / self.hospital_infection_periods)
        h       = self.__convert_to_array(self.hospitalization_fraction)
        d       = self.__convert_to_array(self.community_mortality_fraction)
        d_prime = self.__convert_to_array(self.hospital_mortality_fraction)

        self.exposed_to_infected      = dict(enumerate(σ))
        self.infected_to_resistant    = dict(enumerate((1 - h - d) * γ))
        self.infected_to_hospitalized = dict(enumerate(h * γ))
        self.infected_to_deceased     = dict(enumerate(d * γ))
        self.hospitalized_to_resistant= dict(enumerate((1 - d_prime) * γ_prime))
        self.hospitalized_to_deceased = dict(enumerate(d_prime * γ_prime))

    def set_clinical_parameter(
            self,
            parameter_name,
            values):
        """
        Set a clinical parameter by its name

        Input:
            parameter_name (str): parameter name, like 'latent_periods'
            values (int),
                   (float):    constant value for a parameter
                   (np.array): (self.population,) array of values
        Output:
            None
        """
        setattr(self, parameter_name, values)

    # TODO move this into utilities.py or something
    @singledispatchmethod
    def __convert_to_array(
            self,
            values):
        raise ValueError(
                self.__class__.__name__
                + ": this type of argument is not supported: "
                + values.__class__.__name__)

    @__convert_to_array.register(int)
    @__convert_to_array.register(float)
    def __convert_to_array_const(
            self,
            value):
        return np.full(self.population, value)

    @__convert_to_array.register(np.ndarray)
    def __convert_to_array_array(
            self,
            values):
        return values


