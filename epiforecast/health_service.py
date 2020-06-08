import copy
import numpy as np
import networkx as nx
from functools import singledispatch

from .utilities import normalize, NotInvolving

@singledispatch
def recruit_health_workers(workers, network):
    return workers.recruit(network)

@recruit_health_workers.register(int)
def recruit_health_workers_randomly(workers, network):
    return np.random.choice(network.nodes(), workers)

@recruit_health_workers.register(list)
@recruit_health_workers.register(np.array)
def recruit_health_workers_from_list(workers, network):
    return workers

class HealthService:
    """
    Class to represent the actions of a health service for the population,
    This class is initiated with a full static network of health_workers
    and community known as the `static_population_network`. Patients are
    added from this population by removing their old connectivity and
    adding new contact edges to some `assigned health workers:

    |----------------network------------------|

    [health_worker 1: 'S']---[community 1: 'S']
                          \ /
                           X-[community 2: 'I']
                          / \
    [health_worker 2: 'S']---[community 3: 'H']

    |-----------------------------------------|

    The primary method of this class is:

    population_network = `discharge_and_obtain_patients(current_statuses)`

    The output is a network identified with only human beings, known as the `population network`
    and the connectivity would be given by the following:

    (obtain) Any `community` or `health_worker` node that (and not already a `patient`)
    with status 'H' (hospitalized)  will lose their edges connecting them to current
    neighbours and gain the edges of some health workers. If there is capacity to do so.
    We refer to them as a `patient`.

    (discharge) Any `patient` with status !='H' (i.e it is now resistant 'R' or deceased 'D') will lose
    the edges to their current `health_worker` neighbours and regain thier original neighbours.

    We perform first (discharge) to empy beds, then (obtain) to gain new patients in the function.

    For example. Applying this function to the world network above yields the following.
    1) discharge: There is noone to discharge from beds 1,2
    2) obtain: `community 3` has status H and so is placed into hospital. They lose their community edges
               and gain the edges of `hospital bed 1`. We denote them `patient 1`.

    |---------------------Population network---------------------|

    [Patient 1: 'H']---[health_worker 1: 'S']---[community 1: 'S']
                    \                        \ /
                     \                        /-[community 2: 'I']
                      \                      /
                       [health_worker 2: 'S']

    |------------------------------------------------------------|

    """

    def __init__(self, static_population_network, health_workers, health_workers_per_patient=5):
        """
        Args
        ----
        static_population_network (networkx graph): The full (static) network of possible contacts for
                                                    `health_workers` and `community` nodes.

        health_workers: determines health workers to recruit. Options are:

            * (int): `health_workers` are chosen randomly from the nodes of `static_population_network`
            * (list or array): denotes the address of `health_workers`.
            * (object): picks health workers by executing the function `health_workers.recruit(static_population_network)`

        """
        self.static_population_network = copy.deepcopy(static_population_network)
        self.populace = list(static_population_network.nodes)
        self.population = len(static_population_network)

        # Hire some health workers. recruit_health_workers dispatches on the type of health_workers
        self.health_workers = set(recruit_health_workers(health_workers, static_population_network))
        self.health_workers_per_patient = health_workers_per_patient

        # List of Patient classes
        self.patients = set()

        # List of people unable to go into hospital due to maximum capacity
        self.waiting_list = set()

    def current_patient_addresses(self):
        return set(p.address for p in self.patients)

    def assign_health_workers(self, patient_address, viable_health_workers):
        """
        Assign health workers to a patient.
        """
        if self.health_workers_per_patient < len(viable_health_workers):

            assigned = np.random.choice(list(viable_health_workers),
                                        size = self.health_workers_per_patient,
                                        replace = False)

        else:
            assigned = viable_health_workers

        health_worker_contacts = [ (patient_address, i) for i in assigned ]

        return health_worker_contacts

    def discharge_and_admit_patients(self, statuses, population_network):
        """
        Discharge and admit patients.
        """
        # Discharge recovered and deceased patients first
        discharged_patients = self.discharge_patients(statuses, population_network)

        # Then admit new patients
        admitted_patients = self.admit_patients(statuses, population_network)

        admitted_people = [p.address for p in admitted_patients]
        discharged_people_and_statuses = [(p.address, statuses[p.address]) for p in discharged_patients]
        current_patient_addresses = self.current_patient_addresses()

        print("[ Patient manifest ]          Admitted: ", end='')
        print(*admitted_people, sep=', ')
        print("                            Discharged: ", end='')
        print(*discharged_people_and_statuses, sep=', ')
        print("                               Current: ", end='')
        print(*current_patient_addresses, sep=', ')

        return admitted_patients, discharged_patients

    def discharge_patients(self, statuses, population_network):
        """
        Removes a patient from self.patients and reconnect the patient with their neighbours
        if their status is no longer H.
        """

        discharged_patients = set()
        discharged_community_contacts = []

        for i, patient in enumerate(self.patients):
            if statuses[patient.address] != 'H': # patient is no longer hospitalized
                population_network.remove_edges_from(patient.health_worker_contacts)
                discharged_community_contacts += patient.community_contacts
                discharged_patients.add(patient)

        self.patients = self.patients - discharged_patients

        # Filter contacts with current patients from the list of contacts to add to network
        discharged_community_contacts = [edge for edge in filter(NotInvolving(self.current_patient_addresses()), discharged_community_contacts)]
        population_network.add_edges_from(discharged_community_contacts)

        return discharged_patients

    def admit_patients(self, statuses, population_network):
        """
        Method to find unoccupied beds, and admit patients from the community (storing their details).
        """

        # Set of all hospitalized people
        hospitalized_people = set(i for i in self.populace if statuses[i] == 'H')

        # Hospitalized health workers do not care for patients
        viable_health_workers = self.health_workers - hospitalized_people

        # Patients waiting to be admitted
        waiting_room = hospitalized_people - self.current_patient_addresses()

        admissions = set() # paperwork

        # Admit patients
        for person in waiting_room:

            health_worker_contacts = self.assign_health_workers(person, viable_health_workers)

            community_contacts = list(self.static_population_network.edges(person))

            # Record patient information
            new_patient = Patient(person,
                                  community_contacts,
                                  health_worker_contacts)

            # Admit the patient
            self.patients.add(new_patient)
            admissions.add(new_patient)

            # Rewire the population contact network
            population_network.remove_edges_from(new_patient.community_contacts)
            population_network.add_edges_from(new_patient.health_worker_contacts)

        return admissions




class Patient:
    """
    Patient in hospital and their information.
    """
    def __init__(self, address, community_contacts, health_worker_contacts):
        """
        Args
        ----
        address (int): Location of the patient in the `static_population_network`.

        community_contacts (list of tuples): List of edges in the `static_population_network`.
                                             These are stored here while the patient is in hospital

        health_worker_contacts (list of tuples): a list of edges connecting patient to assigned health workers
        """
        self.address = address
        self.community_contacts = community_contacts
        self.health_worker_contacts = health_worker_contacts
