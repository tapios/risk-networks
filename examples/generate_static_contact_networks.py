import numpy as np
import os, sys; sys.path.append(os.path.join(".."))

from epiforecast.contacts import ContactGenerator, StaticNetworkTimeSeries
from epiforecast.contacts import load_edges

np.random.seed(12345)

edges = load_edges(os.path.join('..', 'data', 'networks', 'edge_list_SBM_1e3.txt'))

fraction_active_edges = 0.034
second = 1 / 60 / 60 / 24
mean_contact_duration = 45 * second # unit: days

# construct the generator
contact_generator = ContactGenerator(edges, fraction_active_edges, mean_contact_duration)

# create the list of static networks (for 1 day)
static_intervals_per_day = int(2) #must be int
static_network_interval = 1.0 / static_intervals_per_day

day_of_contact_networks = StaticNetworkTimeSeries(edges=edges)

day_of_contact_networks.add_networks(
        contact_generator.generate_static_networks(averaging_interval = static_network_interval)
        )

for i in range(len(day_of_contact_networks.time)):
    print("Mean contact duration at t =", 
          day_of_contact_networks.time[i],
          "is",
          np.mean(day_of_contact_networks.contact_networks[i]),
          "days")
