import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from epiforecast.contact_simulator import ContactSimulator, diurnal_inception_rate

np.random.seed(1234)

minute = 1 / 60 / 24

λnight = 3
λday = 40
μ = 1.0 / minute
n_contacts = 10000
dt = 0.1 / 24 # days
days = 10
steps = int(days / dt)

contact_network = nx.barabasi_albert_graph(int(n_contacts / 10), 10)

simulator = ContactSimulator(contact_network = contact_network,
                             mean_event_lifetime = 1 / μ,
                             day_inception_rate = λday,
                             night_inception_rate = λnight,
                             start_time = -dt)

# Generate a time-series of contact durations and average number of active contacts
n_contacts = nx.number_of_edges(contact_network)
contact_durations = np.zeros((steps, n_contacts))
mean_contact_durations = np.zeros(steps)
measurement_times = np.arange(start=0.0, stop=(steps+1)*dt, step=dt)

simulator.run(stop_time = 0.0)

start = timer()

for i in range(steps):

    stop_time = (i + 1) * dt

    if stop_time == 5:
        nx.set_node_attributes(contact_network, values=5, name="day_inception_rate")

    simulator.run(stop_time = stop_time)

    mean_contact_durations[i] = simulator.contact_duration.mean()

    contact_durations[i, :] = simulator.contact_duration

end = timer()

print("Simulated", nx.number_of_edges(contact_network),
      "contacts in {:.3f} seconds".format(end - start))

fig, axs = plt.subplots(nrows=2, figsize=(10, 4), sharex=True)

plt.sca(axs[0])
for i in range(int(np.round(n_contacts/10))):
    plt.plot(measurement_times[1:], contact_durations[:, i] / dt,
             marker=".", mfc="xkcd:light pink", mec=None, alpha=0.02, markersize=0.1)
             
plt.plot(measurement_times[1:], mean_contact_durations / dt,
         linestyle="-", linewidth=3, color="xkcd:sky blue", label="Ensemble mean $ \\langle T_i \\rangle $")
         
plt.ylabel("Specific contact duration, $T_i$")
plt.legend(loc='upper right')

plt.sca(axs[1])
plt.plot(measurement_times[1:], mean_contact_durations / dt,
         linestyle="-", linewidth=1, color="xkcd:sky blue")

plt.xlabel("Time (days)")
plt.ylabel("$ \\langle T_i \\rangle $")

for ax in axs:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axs[0].spines["bottom"].set_visible(False)

plt.show()
