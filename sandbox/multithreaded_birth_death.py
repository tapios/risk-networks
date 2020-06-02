from numba import njit, prange, float64
import numpy as np
from timeit import default_timer as timer

import matplotlib.pyplot as plt

@njit
def diurnal_inception_rate(λmin, λmax, t):
    return np.maximum(λmin, λmax * (1 - np.cos(np.pi * t)**4)**4)

@njit
def accumulate_contact(
                       stop_time,
                       contact_duration, 
                       min_inception_rate,
                       max_inception_rate,
                       mean_contact_lifetime
                      ):

    contact_duration = 0.0
    contact_timeseries = []
    time = []

    inceptions = 0

    while event_time < stop_time

        if contact: # Compute contact deactivation time.

            # Contact is deactivated after
            time_step = - np.log(np.random.random()) * mean_contact_lifetime
            contact_duration += time_step

            # Deactivation
            event_time += time_step
            contact = False

        else: # Compute next contact inception.

            inception_rate = diurnal_inception_rate(min_inception_rate, 
                                                    max_inception_rate,
                                                    event_time)

            # Next inception occurs after:
            time_step = - np.log(np.random.random()) / inception_rate

            # Contact inception
            event_time += time_step
            contact = True
            inceptions += 1

        contact_timeseries.append(contact)
        time.append(event_time)

    return time, contact_timeseries


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

            inception_rate = diurnal_inception_rate(min_inception_rate, 
                                                    max_inception_rate,
                                                    event_time)

            # Next inception occurs after:
            time_step = - np.log(np.random.random()) / inception_rate

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
    n = 1
    minute = 1 / 60 / 24
    second = 1 / 60 / 24

    λmin = 6
    λmax = 0.0 #22 / 10
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

    dt = 3 / 24
    steps = int(1 / dt)

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

    plt.plot(measurement_times[1:], measured_inceptions, 's', alpha=0.4)

    plt.plot(measurement_times,
             np.mean(measured_inceptions) * np.ones(len(measurement_times)),
             linewidth=3.0, alpha=0.6, linestyle='-')

    plt.xlabel("Time (days)")
    plt.ylabel("Inceptions")

    plt.sca(axs[1])
    plt.plot(measurement_times[1:], mean_contact_durations[1:])
    plt.plot(measurement_times, εmin * np.ones(len(measurement_times)), "--")

    plt.xlabel("Time (days)")
    plt.ylabel("Mean contact durations")

    plt.show()
