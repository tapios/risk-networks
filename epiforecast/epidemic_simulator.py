import numpy as np
import networkx as nx

from timeit import default_timer as timer

from .contact_simulator import ContactSimulator
from .kinetic_model_simulator import KineticModel, print_statuses

day = 1
hour = day / 24
minute = hour / 60
second = minute / 60

class EpidemicSimulator:
    def __init__(self, 
                 contact_network,
                 mean_contact_lifetime,
                 contact_inception_rate,
                 transition_rates,
                 community_transmission_rate,
                 hospital_transmission_reduction,
                 static_contact_interval,
                 health_service = None,
                 cycle_contacts = False,
                 start_time = 0.0):

        
        self.contact_network = contact_network

        self.health_service = health_service

        if health_service is not None:
            max_edges = nx.number_of_edges(contact_network) + 5 * health_service.patient_capacity
        else:
            max_edges = nx.number_of_edges(contact_network)

        self.contact_simulator = ContactSimulator(n_contacts = max_edges,
                                                  mean_event_lifetime = mean_contact_lifetime,
                                                  inception_rate = contact_inception_rate,
                                                  start_time = start_time)

        self.kinetic_model = KineticModel(contact_network = contact_network,
                                          transition_rates = transition_rates,
                                          community_transmission_rate = community_transmission_rate,
                                          hospital_transmission_reduction = hospital_transmission_reduction)
                 
        self.static_contact_interval = static_contact_interval

        self.cycle_contacts = cycle_contacts
        self.time = start_time

    def set_statuses(self, statuses):
        self.kinetic_model.set_statuses(statuses)

    def run(self, stop_time, cycle_contacts=None):

        start_run = timer()

        if cycle_contacts is not None:
            self.cycle_contacts = cycle_contacts

        # Duration of the run
        run_time = stop_time - self.time

        # Number of steps
        constant_steps = int(np.floor(run_time / self.static_contact_interval))

        # Interval stop times
        interval_stop_times = self.static_contact_interval * np.arange(start = 1, stop = 1 + constant_steps)

        if self.cycle_contacts:
            print("Generating daily cycle of contacts...")

            self.contact_simulator.set_time(self.time) # Resets the contact simulator
            daily_steps = np.arange(start = 1, stop = 1 + int(1 * day / self.static_contact_interval))
            daily_contacts_cycle = []

            start = timer()

            for step in daily_steps:
                step_time = self.time + step * self.static_contact_interval
                interval_contact_duration = self.contact_simulator.mean_contact_duration(stop_time = step_time)
                daily_contacts_cycle.append(interval_contact_duration)

            end = timer()

            print(" ... done. ({:.2f} s)".format(end - start))

        # Step forward
        for i in range(constant_steps):

            interval_stop_time = interval_stop_times[i]

            if self.health_service is not None:
                self.health_service.discharge_and_admit_patients(self.kinetic_model.current_statuses,
                                                                 self.contact_network)

            start = timer()

            if self.cycle_contacts:
                i = int(np.round(np.mod(interval_stop_time, 1) / self.static_contact_interval))
                contact_duration = daily_contacts_cycle[i]
            else:
                contact_duration = self.contact_simulator.mean_contact_duration(stop_time=interval_stop_time)

            self.kinetic_model.set_mean_contact_duration(contact_duration)
            self.kinetic_model.simulate(self.static_contact_interval)
            self.time += self.static_contact_interval

            end = timer()

            print("Epidemic day: {: 7.3f}, wall_time: {: 6.3f} s,".format(self.kinetic_model.times[-1], end - start),
                  "mean(w_ji): {: 3.0f} min,".format(contact_duration.mean() / minute),
                  "statuses: ",
                  "S {: 4d} |".format(self.kinetic_model.statuses['S'][-1]),
                  "E {: 4d} |".format(self.kinetic_model.statuses['E'][-1]),
                  "I {: 4d} |".format(self.kinetic_model.statuses['I'][-1]),
                  "H {: 4d} |".format(self.kinetic_model.statuses['H'][-1]),
                  "R {: 4d} |".format(self.kinetic_model.statuses['R'][-1]),
                  "D {: 4d} |".format(self.kinetic_model.statuses['D'][-1]))


        if self.time != stop_time: # One final step...
            if self.cycle_contacts:
                self.contact_simulator.set_time(self.time)

            contact_duration = self.contact_simulator.mean_contact_duration(stop_time=stop_time)
            self.kinetic_model.set_mean_contact_duration(contact_duration)
            self.kinetic_model.simulate(stop_time - self.time)
            self.time = stop_time

        end_run = timer()

        print("\n(Epidemic simulation took {:.3f} seconds.)\n".format(end_run-start_run))
