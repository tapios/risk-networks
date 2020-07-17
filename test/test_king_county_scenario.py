# Get the epiforecast package onto our path:
import os, sys; sys.path.append(os.path.join(".."))

import numpy as np

from epiforecast.scenarios import king_county_transition_rates

def get_rates(rates_name):
    return getattr(king_county_transition_rates(population=3, random_seed=1234), rates_name)

transition_rates = king_county_transition_rates(population=3, random_seed=1234)
 
for name in (
              'latent_periods',
              'community_infection_periods',
              'hospital_infection_periods',
              'hospitalization_fraction',
              'community_mortality_fraction',
              'hospital_mortality_fraction',
            ):

    print(get_rates(name))

def test_latent_periods(): 
    assert get_rates('latent_periods')[1] == 6.635600468016812 

def test_community_infection_periods(): 
    assert get_rates('community_infection_periods')[1] == 1.9142295330323456

def test_hospital_infection_periods(): 
    assert get_rates('hospital_infection_periods')[1] == 8.100370683584359 

def test_hospitalization_fraction(): 
    assert get_rates('hospitalization_fraction')[1] == 0.37838525036214415

def test_community_mortality_fraction(): 
    assert get_rates('community_mortality_fraction')[1] == 5.031976849805044e-38
            
def test_hospital_mortality_fraction(): 
    assert get_rates('hospital_mortality_fraction')[1] == 7.305220148909824e-07
