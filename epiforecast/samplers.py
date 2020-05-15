# TODO: What is the correct name for this class?
class ScaledBetaSampler:
    """
    A class representing a parameterized beta distribution.
    It's primary method is `sampler.draw()`, which returns 
    `scale * sample`, where

        `sample ~ Beta(b * p / (1 - p), b)`

    Args
    ----
    
    scale : Number by which to scale `flip`.
        p : TODO: What do we call "p"?
        b : The 'beta' parameter in the Beta distribution.


    This class is used to model the distribution of infection
    rates among a population.
    """
    def __init__(self, scale, p, b):
        self.scale = scale
        self.b = b # "beta" on Wikipedia
        self.p = p # TODO: correctly name this parameter.

    def draw(self):
        """Return scale * scale, where `sample ~ Beta(b * p / (1 - p), b)`"""
        return scale * np.random.beta(b * p / (1 - p), b=b)
