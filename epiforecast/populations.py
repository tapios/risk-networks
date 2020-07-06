import numpy as np
import networkx as nx

from .samplers import AgeDependentBetaSampler, AgeDependentConstant, BetaSampler, GammaSampler

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

    CLINICAL_PARAMETER_NAMES = {
            'latent_periods':               0,
            'community_infection_periods':  1,
            'hospital_infection_periods':   2,
            'hospitalization_fraction':     3,
            'community_mortality_fraction': 4,
            'hospital_mortality_fraction':  5
    }

    def __init__(
            self,
            population,
            lp_sampler,
            cip_sampler,
            hip_sampler,
            hf_sampler,
            cmf_sampler,
            hmf_sampler,
            distributional_parameters=None,
            lp_transform='None',
            cip_transform='None',
            hip_transform='None',
            hf_transform='None',
            cmf_transform='None',
            hmf_transform='None',
            ):
        """
        Constructor with samplers

        For readability, long names are abbreviated using the following
        glossary:
            lp  :   latent periods
            cip :   community infection periods
            hip :   hospital infection periods
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
            distributional_parameters (np.array): (self.population,) array
                                      (None), default: np.ones(self.population)
                                                       is used in this case
        """
        # TODO extract clinical parameters into its own class
        self.population  = population
        self.lp_sampler  = lp_sampler
        self.cip_sampler = cip_sampler
        self.hip_sampler = hip_sampler
        self.hf_sampler  = hf_sampler
        self.cmf_sampler = cmf_sampler
        self.hmf_sampler = hmf_sampler

        if distributional_parameters is None:
            distributional_parameters = np.ones(self.population)

        self.latent_periods               = None
        self.community_infection_periods  = None
        self.hospital_infection_periods   = None
        self.hospitalization_fraction     = None
        self.community_mortality_fraction = None
        self.hospital_mortality_fraction  = None

        self.__draw_and_set_clinical_using(distributional_parameters)

        self.lp_transform  = lp_transform
        self.cip_transform = cip_transform
        self.hip_transform = hip_transform
        self.hf_transform  = hf_transform
        self.cmf_transform = cmf_transform
        self.hmf_transform = hmf_transform

        self.exposed_to_infected       = None
        self.infected_to_resistant     = None
        self.infected_to_hospitalized  = None
        self.infected_to_deceased      = None
        self.hospitalized_to_resistant = None
        self.hospitalized_to_deceased  = None

    # TODO _maybe_ move this into utilities.py or something
    @classmethod
    def __draw_using(
            cls,
            sampler,
            distributional_parameters):
        """
        Draw samples using sampler and its distributional parameters

        Input:
            sampler (int),
                    (float),
                    (np.array): value(s) that are returned unchanged
                    (list): values; transformed into np.array
                    (BetaSampler),
                    (GammaSampler),
                    (AgeDependentBetaSampler),
                    (AgeDependentConstant): samplers to use for drawing
            distributional_parameters (iterable):
                an object used for sampling; redundant in (int), (float),
                (np.array), (list) cases
        Output:
            samples (int),
                    (float): same as `sampler` for (int), (float) cases
                    (np.array): array of samples
        """
        dp = distributional_parameters
        if isinstance(sampler, (int, float, np.ndarray)):
            return cls.__draw_using_const_array(sampler, dp)
        elif isinstance(sampler, list):
            return cls.__draw_using_list(sampler, dp)
        elif isinstance(sampler, (BetaSampler, GammaSampler)):
            return cls.__draw_using_sampler(sampler, dp)
        elif isinstance(sampler, AgeDependentBetaSampler):
            return cls.__draw_using_age_beta_sampler(sampler, dp)
        elif isinstance(sampler, AgeDependentConstant):
            return cls.__draw_using_age_const(sampler, dp)
        else:
            raise ValueError(
                    cls.__class__.__name__
                    + ": this type of argument is not supported: "
                    + sampler.__class__.__name__)

    @staticmethod
    def __draw_using_const_array(parameter, distributional_parameters):
        return parameter

    @staticmethod
    def __draw_using_list(parameter_list, distributional_parameters):
        return np.array(parameter_list)

    @staticmethod
    def __draw_using_sampler(sampler, distributional_parameters):
        return np.array([sampler.draw() for param in distributional_parameters])

    @staticmethod
    def __draw_using_age_beta_sampler(sampler, distributional_parameters):
        return np.array([sampler.draw(param)
                         for param in distributional_parameters])

    @staticmethod
    def __draw_using_age_const(sampler, distributional_parameters):
        return np.array([sampler.constants[param]
                         for param in distributional_parameters])

    def __draw_and_set_clinical_using(
            self,
            distributional_parameters):
        """
        Draw and set clinical parameters using distributional parameters

        Samplers provided in the constructor for clinical parameter (periods and
        fractions) might be either distributions that depend on certain
        parameters (say, mean and variance in the Gaussian case), or arrays, or
        constant values.
        This method uses those samplers to draw from distributions according to
        the provided distributional parameters:
          - constant values and arrays are left unchanged;
          - samplers are used to draw an array of size self.population.

        Note that this method is NOT idempotent, i.e. in the following
            transition_rates.__draw_and_set_clinical_using(dp)
            transition_rates.__draw_and_set_clinical_using(dp)
        second call draws new samples (for those clinical parameters which had
        samplers specified in the constructor).
        It leaves others unchanged though.

        Input:
            distributional_parameters (np.array): (self.population,) array

        Output:
            None
        """
        assert distributional_parameters.shape == (self.population,)

        self.latent_periods               = self.__draw_using(
                self.lp_sampler,
                distributional_parameters)
        self.community_infection_periods  = self.__draw_using(
                self.cip_sampler,
                distributional_parameters)
        self.hospital_infection_periods   = self.__draw_using(
                self.hip_sampler,
                distributional_parameters)
        self.hospitalization_fraction     = self.__draw_using(
                self.hf_sampler,
                distributional_parameters)
        self.community_mortality_fraction = self.__draw_using(
                self.cmf_sampler,
                distributional_parameters)
        self.hospital_mortality_fraction  = self.__draw_using(
                self.hmf_sampler,
                distributional_parameters)

    def transform_clinical_parameter(
            self,
            clinical_parameter,
            transform_type):
        """
        Transforms the clinical parameter by the transform type and returns the output

        Args
        ----
        clinical_parameter(np.array): saved clinical parameter
        transform_type    (string): a string defining the transform implemented :
         - 'None' = no transform required
         - 'log'  = clinical parameter is the logarithm of the desired object, so we exponentiate.
        """

        if transform_type == 'None':
            return clinical_parameter

        elif transform_type == 'log':
            return np.exp(clinical_parameter)

        else:
            raise ValueError("transform_type not recognised, choose from 'None' (default) or 'log' ")

    def add_noise_to_clinical_parameters(
            self,
            parameter_str,
            noise_level):
        """
        Adds Gaussian Noise to the stored clinical_parameter (elementwise)

        Args
        ----
        noise_level (list of Floats): Size of standard deviation of the noise
        parameter_string (list of strings): the parameters to add noise too
        """
        for (lvl,par_str) in zip(noise_level,parameter_str):
            clinical_parameter = self.get_clinical_parameter(par_str)
            noise = np.random.normal(0,lvl,clinical_parameter.shape)
            setattr(self, par_str, clinical_parameter + noise)

    def calculate_from_clinical(self):
        """
        Calculate transition rates using the current clinical parameters

        Output:
            None
        """
        #apply relevant transforms
        lp  = self.transform_clinical_parameter(self.latent_periods,              self.lp_transform)
        cif = self.transform_clinical_parameter(self.community_infection_periods, self.cip_transform)
        hip = self.transform_clinical_parameter(self.hospital_infection_periods,  self.hip_transform)
        hf  = self.transform_clinical_parameter(self.hospitalization_fraction,    self.hf_transform)
        cmf = self.transform_clinical_parameter(self.community_mortality_fraction,self.cmf_transform)
        hmf = self.transform_clinical_parameter(self.hospital_mortality_fraction, self.hmf_transform)

        σ      = self.__broadcast_to_array(1 / lp)
        γ      = self.__broadcast_to_array(1 / cif)
        γ_prime= self.__broadcast_to_array(1 / hip)
        h      = self.__broadcast_to_array(hf)
        d      = self.__broadcast_to_array(cmf)
        d_prime= self.__broadcast_to_array(hmf)

        self.exposed_to_infected      = dict(enumerate(σ))
        self.infected_to_resistant    = dict(enumerate((1 - h - d) * γ))
        self.infected_to_hospitalized = dict(enumerate(h * γ))
        self.infected_to_deceased     = dict(enumerate(d * γ))
        self.hospitalized_to_resistant= dict(enumerate((1 - d_prime) * γ_prime))
        self.hospitalized_to_deceased = dict(enumerate(d_prime * γ_prime))

    def get_clinical_parameters_total_count(self):
        """
        Get the total number of values of all clinical parameters

        Output:
            n_parameters (int): total number of clinical parameters values
        """
        n_parameters = 0
        for name in self.CLINICAL_PARAMETER_NAMES:
            n_parameters += self.get_clinical_parameter_count(name)

        return n_parameters

    def get_clinical_parameter_count(
            self,
            name):
        """
        Get the total number of values of a clinical parameter by its name

        Input:
            name (str): parameter name, like 'latent_periods'
        Output:
            n_parameter (int): total number of the clinical parameter values
        """
        parameter = self.get_clinical_parameter(name)
        if isinstance(parameter, (int, float)):
            return 1
        elif isinstance(parameter, np.ndarray):
            return parameter.size
        else:
            raise ValueError(
                    self.__class__.__name__
                    + ": cannot infer the number of values of parameter: "
                    + name
                    + "; it is of the type: "
                    + values.__class__.__name__)

    def get_clinical_parameters_as_array(self):
        """
        Get values of all clinical parameters as np.array

        The order is the same as specified in CLINICAL_PARAMETER_NAMES.

        Output:
            clinical_parameters (np.array): (n_parameters,) array of values
        """
        clinical_parameters_list = []
        for name in self.CLINICAL_PARAMETER_NAMES:
            clinical_parameters_list.append(self.get_clinical_parameter(name))

        clinical_parameters = np.hstack(clinical_parameters_list)
        return clinical_parameters

    def get_clinical_parameter_indices(
            self,
            name):
        """
        Get indices of a clinical parameter by its name in concatenated array

        The indices are consistent with 'get_clinical_parameters_as_array'
        method, i.e. can be used to get slices of a particular parameter:
            clinical_array = transition_rates.get_clinical_parameters_as_array()
            lp_indices = transition_rates.get_clinical_parameter_indices(
                    'latent_periods')
            latent_periods = clinical_array[lp_indices]

        It is identical to calling:
            latent_periods = transition_rates.get_clinical_parameter(
                    'latent_periods')
        but provides more flexibility (e.g. when storing and accessing
        parameters as arrays)

        Input:
            name (str): parameter name, like 'latent_periods'
        Output:
            indices (np.array): (k,) array of indices
        """
        start_index = 0
        for iteration_name in self.CLINICAL_PARAMETER_NAMES:
            if iteration_name == name:
                break
            start_index += self.get_clinical_parameter_count(iteration_name)

        end_index = start_index + self.get_clinical_parameter_count(name)
        return np.r_[start_index : end_index]

    def get_clinical_parameter(
            self,
            name):
        """
        Get a clinical parameter by its name

        Input:
            name (str): parameter name, like 'latent_periods'
        Output:
            clinical_parameter (int),
                               (float):    constant value of a parameter
                               (np.array): (self.population,) array of values
        """
        return getattr(self, name)

    def set_clinical_parameter(
            self,
            name,
            value):
        """
        Set a clinical parameter by its name

        Input:
            name (str): parameter name, like 'latent_periods'
            value (int),
                  (float):    constant value for a parameter
                  (np.array): (self.population,) array of values
        Output:
            None
        """
        setattr(self, name, value)

    # TODO _maybe_ move this into utilities.py or something
    def __broadcast_to_array(
            self,
            values):
        """
        Broadcast values to numpy array if they are not already

        Input:
            values (int),
                   (float): constant value to be broadcasted
                   (np.array): (population,) array of values

        Output:
            values_array (np.array): (population,) array of values
        """
        if isinstance(values, (int, float)):
            return self.__broadcast_to_array_const(values)
        elif isinstance(values, np.ndarray):
            return self.__broadcast_to_array_array(values)
        else:
            raise ValueError(
                    self.__class__.__name__
                    + ": this type of argument is not supported: "
                    + values.__class__.__name__)

    def __broadcast_to_array_const(
            self,
            value):
        return np.full(self.population, value)

    def __broadcast_to_array_array(
            self,
            values):
        return values


