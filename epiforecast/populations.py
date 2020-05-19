import numpy as np

class ClinicalStatistics:
    def __init__(self, ages, minimum=0, sampler=None):

        if sampler is not None:
            self.values = np.array([minimum + sampler.draw(age) for age in ages])
        else:
            self.values = np.array([minimum for age in ages])

        self.population = len(ages)
        self.ages = ages
        self.minimum = minimum

def populate_ages(population=1000, distribution=[0.25, 0.5, 0.25]):
    classes = len(distribution)

    ages = [np.random.choice(np.arange(classes), p=distribution) 
            for person in range(population)]

    return np.array(ages)
