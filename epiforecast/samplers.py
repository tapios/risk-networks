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

    def draw(self, *args):
        """Return `sample`, where `sample ~ Gamma(k, theta)`"""
        return np.random.gamma(self.k, self.theta)
