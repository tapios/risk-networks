import os, sys

sys.path.append(os.getenv('EPIFORECAST', os.path.join("..")))

import numpy as np
import networkx as nx
import epiforecast
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

# Load a sample contact network
contact_network = epiforecast.load_sample_contact_network()

# Instantiate a model on the sample network
model = epiforecast.StaticRiskNetworkModel(contact_network)

# Set to default transition rates
model.set_transition_rates(epiforecast.TransitionRates())

# Initialize a state with a small, randomly-selected number of infected.
epiforecast.random_infection(model, size=10)

# Integrate the model forwards
time_interval = 0.1
t, states = [], []

for i in range(10):
    residual = model.integrate_forwards(time_interval)
    t.append(model.time)
    states.append(model.state.copy())

t = np.array(t)

S, E, I, R, H, D = epiforecast.unpack_state_timeseries(model, states)

mean_S = S.mean(axis=1)
mean_E = E.mean(axis=1)
mean_I = I.mean(axis=1)
mean_H = H.mean(axis=1)
mean_R = R.mean(axis=1)
mean_D = D.mean(axis=1)


plt.plot(t, mean_S, label="$ \\bar{ S } $")
plt.plot(t, mean_E, label="$ \\bar{ E } $")
plt.plot(t, mean_I, label="$ \\bar{ I } $")
plt.plot(t, mean_H, label="$ \\bar{ H } $")
plt.plot(t, mean_R, label="$ \\bar{ R } $")
plt.plot(t, mean_D, label="$ \\bar{ D } $")

plt.legend()
plt.xlabel("Time")
plt.ylabel("Network-averaged state")
plt.title("Epidemic evolution on a simple static network")

plt.show()
