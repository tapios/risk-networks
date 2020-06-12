import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer

import networkx as nx
import numpy as np
import pandas as pd
import random
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from numba import set_num_threads

set_num_threads(1)

from epiforecast.scenarios import random_epidemic
from epiforecast.network_storage import StaticNetworkSeries
#
# Set random seeds for reproducibility
#
seed = 212212
np.random.seed(seed)

contact_network = nx.barabasi_albert_graph(30000, 10)
population = len(contact_network)

time = 0.0
static_contact_interval = 0.25
simulation_length = 1

contact_network = nx.barabasi_albert_graph(1000, 10)

network_storage = StaticNetworkSeries(static_contact_interval)


statuses = random_epidemic(contact_network,
                           fraction_infected=0.01)

print("saving all the networks")
current_infected = [node for node in contact_network.nodes if statuses[node] == 'I']
print("infected at time", time, current_infected)

for i in range(int(simulation_length/static_contact_interval)):
    #save network and start time statuses
    network_storage.save_network_by_start_time(contact_network=contact_network, start_time=time)
    network_storage.save_start_statuses_to_network(start_time=time, start_statuses=statuses)
    
    #pretend we 'simuate' forward
    contact_network = nx.barabasi_albert_graph(1000, 10)
    statuses = random_epidemic(contact_network, fraction_infected=0.01)
    time = time+static_contact_interval
    current_infected = [node for node in contact_network.nodes if statuses[node] == 'I']
    print("infected at time", time, current_infected)

    #save end time statuses
    network_storage.save_end_statuses_to_network(end_time=time, end_statuses = statuses)

print(" ")
print("loading the networks backwards by end time")
for i in range(int(simulation_length/static_contact_interval)):
    net=network_storage.get_network_from_end_time(end_time=time)
    current_infected = [node for node in net.contact_network.nodes if net.end_statuses[node] == 'I']
    print("infected at time", net.end_time, current_infected)
    time = time - static_contact_interval
                        
print(" ")
print("loading the networks forwards by start time")
for i in range(int(simulation_length/static_contact_interval)):
    net=network_storage.get_network_from_start_time(start_time=time)
    current_infected = [node for node in net.contact_network.nodes if net.start_statuses[node] == 'I']
    print("infected at time", net.start_time, current_infected)
    time = time + static_contact_interval
