import os, sys; sys.path.append(os.path.join(".."))

from epiforecast.temporal_adjacency import TemporalAdjacency
from epiforecast.kinetic_model_simulator import KineticModel



edge_list_file = os.path.join('..', 'data', 'networks', 'edge_list_SBM_1e3.txt')
active_edge_list_frac = 0.034
mean_contact_duration = 1.0 / 1920.0  # unit: days

# construct the generator
contact_generator= TemporalAdjacency(edges_filename,
                                     active_edge_list_frac,
                                     mean_contact_duration)


population=
# create the list of static networks (for 1 day)
static_intervals_per_day = int(8) #must be int
static_network_interval = 1.0/static_intervals_per_day
contact_generator.generate_static_networks(dt_averaging=static_network_interval)

#pathology parameters

age_distribution = [ 0.23,  # 0-19 years
                     0.39,  # 20-44 years
                     0.25,  # 45-64 years
                     0.079  # 65-75 years
                    ]
age_distribution.append(1 - sum(age_distribution))
ages = populate_ages(population, distribution=age_distribution)
# Next, we randomly generate clinical properties for our example population.
# Note that the units of 'periods' are days, and the units of 'rates' are 1/day.
latent_periods              = sample_pathological_distribution(GammaSampler(k=1.7, theta=2.0), population=population, minimum=2)
community_infection_periods = sample_pathological_distribution(GammaSampler(k=1.5, theta=2.0), population=population, minimum=1)
hospital_infection_periods  = sample_pathological_distribution(GammaSampler(k=1.5, theta=3.0), population=population, minimum=1)

hospitalization_fraction     = sample_pathological_distribution(AgeAwareBetaSampler(mean=[ 0.02,  0.17,  0.25, 0.35, 0.45], b=4), ages=ages)
community_mortality_fraction = sample_pathological_distribution(AgeAwareBetaSampler(mean=[0.001, 0.001, 0.005, 0.02, 0.05], b=4), ages=ages)
hospital_mortality_fraction  = sample_pathological_distribution(AgeAwareBetaSampler(mean=[0.001, 0.001,  0.01, 0.04,  0.1], b=4), ages=ages)

# We process the clinical data to determine transition rates between each epidemiological state,

transition_rates = TransitionRates(population,
                                   latent_periods,
                                   community_infection_periods,
                                   hospital_infection_periods,
                                   hospitalization_fraction,
                                   community_mortality_fraction,
                                   hospital_mortality_fraction)

transition_rates = TransitionRates(...)
transmission_rate = 0.1

kinetic_model = KineticModel(day_of_contact_networks[0]
                             transition_rates,
                             transmission_rate,
                             hospital_transmission_reduction)

kinetic_states=np.zeros([static_intervals_per_day,6*population])
for i range(static_intervals_per_day):
    Kinetic_model.update_contact_network(day_of_contact_networks[i]) 
   
    Kinetic_model.simulate(static_contact_interval) #simulate from 0 until interval

    #If we did DA, would also use:
    #Kinetic_model.update_transmission_rate(transmission_rate) # update const beta if required
    #kinetic_model.update_transition_rates(transition_rates) #update nodal rates based on object (see alfredo's implementation in master_equations)
