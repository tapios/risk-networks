from numba import njit, prange, float64
import numpy as np
from timeit import default_timer as timer

from scipy.special import roots_legendre

import matplotlib.pyplot as plt

# See discussion in 
#
# Christian L. Vestergaard , Mathieu Génois, "Temporal Gillespie Algorithm: Fast Simulation 
# of Contagion Processes on Time-Varying Networks", PLOS Computational Biology (2015)
#
# https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004579

# Generate Gaussian quadrature weights and nodes as global variables
N = 4
x, w = roots_legendre(N)
x = (x + 1) / 2
w = w / 2

@njit 
def integrate_inception_rate(λmin, λmax, t, Δt):
    integral = 0.0
    for i in range(N):
        integral += w[i] * diurnal_inception_rate(λmin, λmax, t + x[i] * Δt)

    return integral

@njit
def diurnal_inception_rate(λmin, λmax, t):
    return np.maximum(λmin, λmax * (1 - np.cos(np.pi * t)**4)**4)

@njit
def simulate_contact(
                     start_time,
                     stop_time,
                     event_time,
                     contact_duration, 
                     overshoot_duration,
                     contact,
                     min_inception_rate,
                     max_inception_rate,
                     mean_contact_lifetime
                    ):

    contact_duration = overshoot_duration

    inceptions = 0

    while event_time < stop_time:

        if contact: # Compute contact deactivation time.

            # Contact is deactivated after
            time_step = - np.log(np.random.random()) * mean_contact_lifetime
            contact_duration += time_step

            # Deactivation
            event_time += time_step
            contact = False

        else: # Compute next contact inception.

            # Normalized step with τ ~ Exp(1)
            τ = - np.log(np.random.random())

            # Initial guess for waiting time
            inception_rate = diurnal_inception_rate(min_inception_rate, 
                                                    max_inception_rate,
                                                    event_time)

            time_step = τ / inception_rate

            # Iteratively solve for waiting time. Integral over time-dependent
            # rate is computed with Gaussian quadrature.
            # 
            # Note: 
            #     - Iterative solve may not be stable.
            #     - See citation at top of file for more information.
            #
            for i in range(2):
                time_step = τ / integrate_inception_rate(min_inception_rate,
                                                         max_inception_rate,
                                                         event_time,
                                                         time_step)

            # Contact inception
            event_time += time_step
            contact = True
            inceptions += 1

    # We 'overshoot' the end of the interval. To account for this, we subtract the
    # overshoot, and the contribution of the 'overshoot' to the total contact duration.
    #
    #              < -------------------- >
    #                     overshoot 
    #
    #             stop
    # ----- x ---- | -------------------- x
    #     prev                          next
    #     step                          step
    #
    # A confusing part of this algorithm is that the *current* state is the inverse
    # of the prior state. Therefore if we are *currently* in contact, we were *not*
    # in contact during the overshoot interval --- and vice versa. This is why
    # we write (1 - contact) below.

    overshoot_duration = (event_time - stop_time) * (1 - contact)
    contact_duration -= overshoot_duration

    return event_time, contact, contact_duration, overshoot_duration, inceptions

@njit(parallel=True)
def simulate_contacts(
                      start_time,
                      stop_time,
                      event_time,
                      contact_duration,
                      overshoot_duration,
                      contact,
                      min_inception_rate,
                      max_inception_rate,
                      mean_contact_lifetime,
                      inceptions
                     ):

    for i in prange(len(contact_duration)):

        (event_time[i],
         contact[i],
         contact_duration[i],
         overshoot_duration[i],
         inceptions[i]         ) = simulate_contact(
                                                    start_time,
                                                    stop_time,
                                                    event_time[i],
                                                    contact_duration[i],
                                                    overshoot_duration[i],
                                                    contact[i],
                                                    min_inception_rate[i],
                                                    max_inception_rate[i],
                                                    mean_contact_lifetime[i]
                                                   )
                         

if __name__ == "__main__":

    n = 1000000
    minute = 1 / 60 / 24

    λmin = 3
    λmax = 22
    μ = 1 / minute

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
                          event_time,
                          contact_duration,
                          overshoot_duration,
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
