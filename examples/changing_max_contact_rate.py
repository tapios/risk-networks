import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from epiforecast.contact_simulator import ContactSimulator, diurnal_inception_rate

np.random.seed(1234)

minute = 1 / 60 / 24

λnight = 3
λday = 22
μ = 1.0 / minute
n_contacts = 100000
dt = 1 / 24 # days
days = 7
steps = int(7 / dt)

contact_network = nx.barabasi_albert_graph(int(n_contacts / 10), 10)

simulator = ContactSimulator(contact_network = contact_network,
                             mean_event_lifetime = 1 / μ,
                             day_inception_rate = λday,
                             night_inception_rate = λnight,
                             start_time = -dt)

# Generate a time-series of contact durations and average number of active contacts
contact_durations = np.zeros((steps, 4))
measured_inceptions = np.zeros((steps, 4))
mean_contact_durations = np.zeros(steps)
measurement_times = np.arange(start=0.0, stop=(steps+1)*dt, step=dt)

simulator.run(stop_time = 0.0)

start = timer()

for i in range(steps):

    stop_time = (i + 1) * dt

    if stop_time == 3:
        λday = 5
        nx.set_node_attributes(contact_network, values=λday, name="day_inception_rate")

    simulator.run(stop_time = (i + 1) * dt)

    mean_contact_durations[i] = simulator.contact_duration.mean() / dt

    for j in range(contact_durations.shape[1]):
        contact_durations[i, j] = simulator.contact_duration[j] / dt

end = timer()

print("Simulated", nx.number_of_edges(contact_network),
      "contacts in {:.3f} seconds".format(end - start))

fig, axs = plt.subplots(nrows=2, figsize=(14, 8), sharex=True)

plt.sca(axs[0])
for j in range(contact_durations.shape[1]):
    plt.plot(measurement_times[1:], contact_durations[:, j], '.', alpha=0.6,
             label="Contact {}".format(j))

plt.ylabel("Mean contact durations, $T_i$")
plt.legend(loc='upper right')

plt.sca(axs[1])

t = measurement_times
λ = np.zeros(steps)
for i in range(steps):
    λ[i] = diurnal_inception_rate(λnight, λday, 1/2 * (t[i] + t[i+1]))

plt.plot(t[1:], λ / (λ + μ), linestyle="-", linewidth=3, alpha=0.4, label="$ \lambda(t) / [ \mu + \lambda(t) ] $")

plt.plot(measurement_times[1:], mean_contact_durations, linestyle="--", color="k", linewidth=1,
         label="$ \\bar{T}_i(t) $")

plt.xlabel("Time (days)")
plt.ylabel("Ensemble-averaged $T_i$")
plt.legend(loc='upper right')

plt.show()
