import numpy as np

class AgeAwareBetaSampler:
    """
    Represents a parameterized 'age-aware' Beta distribution.
    It's primary method is `sampler.draw(age)`, which returns `sample`
    such that

        `sample ~ Beta(b[age] * mean[age] / (1 - mean[age]), b[age])`

    Args
    ----
       b : The 'beta' parameter in the Beta distribution.
    mean : The mean of the Beta distribution.

    This class is used to model the distribution of infection
    rates among a population.
    """
    def __init__(self, b, mean=0):

        try:
            self.age_classes = len(mean)
        except TypeError:
            self.age_classes = len(b)
        except:
            print("One of 'mean' or 'b' must be a list.")

        # Convert to numpy arrays of correct length
        self.b = np.ones(self.age_classes) * b       # "beta" on Wikipedia
        self.mean = np.ones(self.age_classes) * mean # Mean value of the beta distribution

    def draw(self, age):
        """Return `sample`, where `sample ~ Beta(b * p / (1 - p), b)`"""
        return np.random.beta(self.b[age] * self.mean[age] / (1 - self.mean[age]), 
                              b=self.b[age])

class BetaSampler:
    """
    Represents a parameterized 'age-aware' Beta distribution.
    It's primary method is `sampler.draw(age)`, which returns `sample`
    such that

        `sample ~ Beta(b * mean / (1 - mean), b)`

    Args
    ----
       b : The 'beta' parameter in the Beta distribution.
    mean : The mean of the Beta distribution.

    This class is used to model the distribution of infection
    rates among a population.
    """
    def __init__(self, b, mean=0):

        # Convert to numpy arrays of correct length
        self.b =  b       # "beta" on Wikipedia
        self.mean = mean # Mean value of the beta distribution

    def draw(self):
        """Return `sample`, where `sample ~ Beta(b * p / (1 - p), b)`"""
        return np.random.beta(self.b * self.mean / (1 - self.mean), b=self.b)


    
class GammaSampler:
    """
    A class representing a parameterized Gamma distribution.
    It's primary method is `sampler.draw(*args)`, which returns 
    `sample`, where

        `sample ~ Gamma(k, theta)`

    Args
    ----
        k : Shape parameter
    theta : Scale parameter

    See https://en.wikipedia.org/wiki/Gamma_distribution.

    This class is used to model the distribution of clinical
    rates (latent period of infection, infectiousness duration) among a population.
    """
    def __init__(self, k, theta):
        self.k = k # shape parameter
        self.theta = theta # scale parameter

    def draw(self):
        """Return `sample`, where `sample ~ Gamma(k, theta)`"""
        return np.random.gamma(self.k, self.theta)


class AgeAwareConstantSampler:
    """
    A class representing a constant distribution.
    It's primary method is `sampler.draw(*args)`, which returns 
    `sample`, where

        `sample ~ constants[age]`

    """
    def __init__(self, constants):
        # input arg is a list of constants for each age category.
        self.constants = constants
      
    def draw(self, age):
        """Return `sample`, where `sample ~ constants[age]`"""
        return self.constants[age]


class ConstantSampler:
    """
    A class representing a constant distribution.
    It's primary method is `sampler.draw(*args)`, which returns 
    `sample`, where

        `sample ~ constant`

    """
    def __init__(self, constant):
        # input arg is a list of constants for each age category.
        self.constant = constant
      
    def draw(self):
        """Return `sample`, where `sample ~ constant category`"""
        return self.constant
