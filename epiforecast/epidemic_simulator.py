import numpy as np
import networkx as nx

from timeit import default_timer as timer

from .contact_simulator import ContactSimulator
from .kinetic_model_simulator import KineticModel, print_statuses
from .utilities import NotInvolving

day = 1
hour = day / 24
minute = hour / 60
second = minute / 60

class EpidemicSimulator:
    def __init__(self,
                 contact_network,
                 transition_rates,
                 community_transmission_rate,
                 hospital_transmission_reduction,
                 static_contact_interval,
                 mean_contact_lifetime,
                 day_inception_rate = None,
                 night_inception_rate = None,
                 health_service = None,
                 start_time = 0.0):

        self.contact_network = contact_network
        self.health_service = health_service

        contacts_buffer = 0

        if health_service is None:
            buffer_margin = 1
        else:
            buffer_margin = 1.2

        self.contact_simulator = ContactSimulator(contact_network,
                                                    day_inception_rate = day_inception_rate,
                                                  night_inception_rate = night_inception_rate,
                                                   mean_event_lifetime = mean_contact_lifetime,
                                                       buffer_margin = buffer_margin,
                                                            start_time = start_time)

        self.kinetic_model = KineticModel(contact_network = contact_network,
                                          transition_rates = transition_rates,
                                          community_transmission_rate = community_transmission_rate,
                                          hospital_transmission_reduction = hospital_transmission_reduction)

        self.static_contact_interval = static_contact_interval
        self.time = start_time

    def run(self, stop_time):

        # Duration of the run
        run_time = stop_time - self.time

        # Number of steps
        constant_steps = int(np.floor(run_time / self.static_contact_interval))

        # Interval stop times
        interval_stop_times = self.time + self.static_contact_interval * np.arange(start = 1, stop = 1 + constant_steps)

        start_run = timer()

        # Step forward
        for i in range(constant_steps):

            interval_stop_time = interval_stop_times[i]

            print("")
            print("")
            print("                               *** Day: {:.3f}".format(interval_stop_time))

            # Manage hospitalization
            start_health_service_action = timer()

            if self.health_service is not None:
                admitted, discharged = self.health_service.discharge_and_admit_patients(self.kinetic_model.current_statuses,
                                                                                        self.contact_network)

                # Find edges to add and remove from contact simulation
                edges_to_remove = set()
                edges_to_add = set()

                current_patients = self.health_service.current_patient_addresses()
                previous_patients = current_patients - {p.address for p in admitted}
                previous_patients.update(p.address for p in discharged)

                if len(admitted) > 0:
                    for patient in admitted:
                        edges_to_remove.update(filter(NotInvolving(previous_patients), patient.community_contacts))
                        edges_to_add.update(patient.health_worker_contacts)

                if len(discharged) > 0:
                    for patient in discharged:
                        edges_to_remove.update(patient.health_worker_contacts)
                        edges_to_add.update(filter(NotInvolving(current_patients), patient.community_contacts))

            else:
                edges_to_add, edges_to_remove = set(), set()

            end_health_service_action = timer()

            # Simulate contacts
            start_contact_simulation = timer()

            self.contact_simulator.run_and_set_edge_weights(stop_time = interval_stop_time,
                                                            edges_to_add = edges_to_add,
                                                            edges_to_remove = edges_to_remove)

            end_contact_simulation = timer()

            # Run the kinetic simulation
            start_kinetic_simulation = timer()

            self.kinetic_model.simulate(self.static_contact_interval)
            self.time += self.static_contact_interval

            end_kinetic_simulation = timer()

            print("[ Status report ]          Susceptible: {:d}".format(self.kinetic_model.statuses['S'][-1]))
            print("                               Exposed: {:d}".format(self.kinetic_model.statuses['E'][-1]))
            print("                              Infected: {:d}".format(self.kinetic_model.statuses['I'][-1]))
            print("                          Hospitalized: {:d}".format(self.kinetic_model.statuses['H'][-1]))
            print("                             Resistant: {:d}".format(self.kinetic_model.statuses['R'][-1]))
            print("                              Deceased: {:d}".format(self.kinetic_model.statuses['D'][-1]))
            print("[ Wall times ]    Hosp. administration: {:.4f} s,".format(end_health_service_action - start_health_service_action))
            print("                    Contact simulation: {:.4f} s,".format(end_contact_simulation - start_contact_simulation))
            print("                    Kinetic simulation: {:.4f} s,".format(end_kinetic_simulation - start_kinetic_simulation))
            print("")

        if self.time < stop_time: # One final step...

            contact_duration = self.contact_simulator.mean_contact_duration(stop_time=stop_time)
            self.kinetic_model.set_mean_contact_duration(contact_duration)
            self.kinetic_model.simulate(stop_time - self.time)
            self.time = stop_time

        end_run = timer()

        print("\n(Epidemic simulation took {:.3f} seconds.)\n".format(end_run-start_run))

    def set_statuses(self, statuses):
        self.kinetic_model.set_statuses(statuses)
