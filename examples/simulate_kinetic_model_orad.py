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
