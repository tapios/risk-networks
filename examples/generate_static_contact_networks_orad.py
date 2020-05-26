import os, sys; sys.path.append(os.path.join(".."))

from epiforecast.temporal_adjacency import TemporalAdjacency



edge_list_file = os.path.join('..', 'data', 'networks', 'edge_list_SBM_1e3.txt')
active_edge_list_frac = 0.034
mean_contact_duration = 1.0 / 1920.0  # unit: days

# construct the generator
contact_generator= TemporalAdjacency(edges_filename,
                                     active_edge_list_frac,
                                     mean_contact_duration)


# create the list of static networks (for 1 day)
static_intervals_per_day = int(8) #must be int
static_network_interval = 1.0/static_intervals_per_day
day_of_contact_networks=contact_generator.generate_static_networks(dt_averaging=static_network_interval)

