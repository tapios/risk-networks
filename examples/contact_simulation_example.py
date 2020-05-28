import os, sys; sys.path.append(os.path.join(".."))

import numpy as np
import matplotlib.pyplot as plt

from epiforecast.contact_simulator import ContactSimulator

np.random.seed(1234)

fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(9, 6))

def plot_contact_simulation(n_contacts, time_step=1/24):

    simulator = ContactSimulator(n_contacts = n_contacts, 
                                 initial_fraction_active_contacts = 0.1)

    network_averaged_contact_duration = []
    active_contacts = []
    times = []
    
    initial_active_contacts = np.count_nonzero(simulator.active_contacts)
    
    # Generate a time-series of contact durations and average number of 
    # active contacts
    for i in range(int(2 / time_step)):
        simulator.simulate_contact_duration(stop_time = (i + 1) * time_step)
    
        mean_interval_contact_duration = simulator.contact_duration.mean()
    
        network_averaged_contact_duration.append(mean_interval_contact_duration)
        times.append(i * time_step)
        active_contacts.append(np.count_nonzero(simulator.active_contacts))
    
    # Store results
    network_averaged_contact_duration = np.array(network_averaged_contact_duration)
    times = np.array(times)
    active_contacts = np.array(active_contacts)

    # Plot
    label = "$ n_{{\mathrm{{contacts}}}} = {} $".format(n_contacts)
    minute = 1 / 60 / 24

    plt.sca(axs[0])
    plt.plot(times, active_contacts, "-", label=label)

    plt.sca(axs[1])
    plt.plot(times, network_averaged_contact_duration / minute, "-", label=label)

    return times, network_averaged_contact_duration, active_contacts

# Loop over a bunch of network sizes
for n_contacts in (1000, 2000, 5000, 10000):
    plot_contact_simulation(n_contacts)

# Format a plot prettily
plt.sca(axs[0])
plt.ylabel("Number of active contacts")
plt.legend()

plt.sca(axs[1])
plt.xlabel("Time (days)")
plt.ylabel("Mean contact duration (minutes)")

plt.savefig("contact_simulation.png", dpi=480)
plt.show()
