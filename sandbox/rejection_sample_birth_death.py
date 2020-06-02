from numba import njit, prange, float64
import numpy as np
from timeit import default_timer as timer

from scipy.special import roots_legendre

import matplotlib.pyplot as plt

@njit
def diurnal_inception_rate(λmin, λmax, t):
    return np.maximum(λmin, λmax * (1 - np.cos(np.pi * t)**4)**4)

@njit
def simulate_contact(
                     time,
                     time_step,
                     n_steps,
                     contact_duration, 
                     contact,
                     min_inception_rate,
                     max_inception_rate,
                     mean_contact_lifetime
                    ):

    # Initialize
    contact_duration = 0.0
    inceptions = 0
    step = 0

    while step < n_steps:

        # Advance to t = i * time_step
        step += 1
        time += time_step

        # Determine whether an event has occurred
        r = np.random.random()

        if contact: # deactivate?
            contact_duration += time_step

            if r < time_step / mean_contact_lifetime:
                contact = False
    
        else: # inception?

            inception_rate = diurnal_inception_rate(min_inception_rate, max_inception_rate, time)

            if r < time_step * inception_rate:
                contact = True
                inceptions += 1

    return contact, contact_duration, inceptions

@njit(parallel=True)
def simulate_contacts(
                      time,
                      stop_time,
                      time_step,
                      contact_duration,
                      contact,
                      min_inception_rate,
                      max_inception_rate,
                      mean_contact_lifetime,
                      inceptions
                     ):

    n_steps = int(np.round((stop_time - time) / time_step))

    for i in prange(len(contact_duration)):

         contact[i], contact_duration[i], inceptions[i] = simulate_contact(
                                                                           time,
                                                                           time_step,
                                                                           n_steps,
                                                                           contact_duration[i],
                                                                           contact[i],
                                                                           min_inception_rate[i],
                                                                           max_inception_rate[i],
                                                                           mean_contact_lifetime[i]
                                                                          )
                         

if __name__ == "__main__":

    n = 1000000
    second = 1 / 60 / 60 / 24
    minute = 60 * second

    λmin = 3
    λmax = 22
    μ = 1 / minute
    time_step = 10 * second

    εmin = λmin / (μ + λmin)
    εmax = λmax / (μ + λmax)

    print("Equilibrium solution:", εmin)

    min_inception_rate = λmin * np.ones(n)
    max_inception_rate = λmax * np.ones(n)
    mean_contact_lifetime = 1 / μ * np.ones(n)

    event_time = np.zeros(n)
    inceptions = np.zeros(n)
    contact_duration = np.zeros(n)
    overshoot_duration = np.zeros(n)
    contacts = np.random.choice([False, True], size = n, p = [1 - εmin, εmin])

    dt = 1 / 24
    steps = int(7 / dt)

    start = timer()

    mean_contact_durations = []
    active_contacts = []
    measurement_times = []
    measured_inceptions = []

    active_contacts.append(np.count_nonzero(contacts))
    mean_contact_durations.append(contact_duration.mean())
    measurement_times.append(0.0)

    for i in range(steps):

        simulate_contacts(
                          i * dt,
                          (i + 1) * dt,
                          time_step,
                          contact_duration,
                          contacts,
                          min_inception_rate,
                          max_inception_rate,
                          mean_contact_lifetime,
                          inceptions
                         )

        fraction_active_contacts = np.count_nonzero(~contacts) / n
        active_contacts.append(np.count_nonzero(~contacts))
        mean_contact_durations.append(contact_duration.mean() / dt)
        measurement_times.append((i + 1) * dt)
        measured_inceptions.append(inceptions[0])

    end = timer()

    print("Simulation time:", end - start)
    print("Mean contact duration:", contact_duration.mean())

    fig, axs = plt.subplots(nrows=2)

    plt.sca(axs[0])

    plt.plot(measurement_times[1:], np.array(measured_inceptions) / dt, '.', alpha=0.4,
             label="in interval")

    plt.plot(measurement_times,
             np.mean(measured_inceptions) * np.ones(len(measurement_times)) / dt,
             linewidth=3.0, alpha=0.6, linestyle='-',
             label="mean")

    plt.plot(measurement_times, 11 * np.ones(len(measurement_times)),
             linewidth=2.0, alpha=0.6, color='k', linestyle='-', label="11 $ \mathrm{day^{-1}} $")

    plt.xlabel("Time (days)")
    plt.ylabel("Measured single-contact inception rate $ \mathrm{(day^{-1})} $")
    plt.legend()

    plt.sca(axs[1])
    plt.plot(measurement_times[1:], mean_contact_durations[1:], label="Contact durations, averaged over contacts")

    plt.plot(measurement_times, εmin * np.ones(len(measurement_times)), "--",
             label="$ \lambda_\mathrm{min} / (\mu + \lambda_\mathrm{min}) $")

    plt.plot(measurement_times, εmax * np.ones(len(measurement_times)), ":",
             label="$ \lambda_\mathrm{max} / (\mu + \lambda_\mathrm{max}) $")

    plt.plot(measurement_times, np.mean(mean_contact_durations[1:]) * np.ones(len(measurement_times)), "k-",
             label="Contact durations, averaged over contacts and simulation interval")

    plt.xlabel("Time (days)")
    plt.ylabel("Mean contact durations")
    plt.legend()

    plt.show()
