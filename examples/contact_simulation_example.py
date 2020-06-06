import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from epiforecast.contact_simulator import ContactSimulator

np.random.seed(1234)

fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(9, 6))

def plot_contact_simulation(contact_network, time_step = 0.1 * 1/24):

    simulator = ContactSimulator(contact_network = contact_network,
                                 mean_event_lifetime = 1 / 60 / 24,
                                 day_inception_rate = 22,
                                 night_inception_rate = 3)

    mean_contact_duration = []
    active_contacts = []
    times = []
    
    initial_active_contacts = np.count_nonzero(simulator.active_contacts)
    
    # Generate a time-series of contact durations and average number of active contacts
    days = 2

    for i in range(int(days / time_step)):
        simulator.run(stop_time = (i + 1) * time_step)
    
        mean_contact_duration.append(simulator.contact_duration.mean())

        times.append(i * time_step)

        active_contacts.append(np.count_nonzero(~simulator.active_contacts))
    
    # Store results
    mean_contact_duration = np.array(mean_contact_duration)
    times = np.array(times)
    active_contacts = np.array(active_contacts)

    # Plot
    label = "$ n_{{\mathrm{{contacts}}}} = {} $".format(n_contacts)
    minute = 1 / 60 / 24

    plt.sca(axs[0])
    plt.plot(times, active_contacts, "-", label=label)

    plt.sca(axs[1])
    plt.plot(times, mean_contact_duration / minute, "-", alpha=0.5, label=label)

    return times, mean_contact_duration, active_contacts

# Loop over a bunch of network sizes
for n_contacts in (5000, 10000, 20000):

    contact_network = nx.barabasi_albert_graph(int(n_contacts / 10), 10)

    start = timer()
    plot_contact_simulation(contact_network)
    end = timer()

    print("Simulated", nx.number_of_edges(contact_network),
          "contacts in {:.3f} seconds".format(end - start))

# Format a plot prettily
plt.sca(axs[0])
plt.ylabel("Number of active contacts")
plt.legend()

plt.sca(axs[1])
plt.xlabel("Time (days)")
plt.ylabel("Mean contact duration (minutes)")

plt.savefig("contact_simulation.png", dpi=480)
plt.show()
