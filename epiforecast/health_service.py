import copy
import numpy as np
import networkx as nx

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

    def __init__(self, patient_capacity, health_worker_population, static_population_network, edges):
        """
        Args
        ----
        static_population_network (networkx graph): This is the full (static) network of possible contacts for
                                                    `health_workers` and `community` nodes

                                       
        """
        self.static_population_network=copy.deepcopy(static_population_network)
        self.populace = copy.deepcopy(list(static_population_network.nodes))
        self.population = len(static_population_network)
        
        self.patient_capacity=patient_capacity
        self.health_workers=list(static_population.nodes)[:health_worker_population]
        
        self.edges=edges
        # Stores a list of Patient classes
        self.patients = []
        # Stores a list of people unable to go into hospital due to maximum capacity 
        self.waiting_list = []

    def assign_health_workers(self, patient_address, num_assigned=5):

        hospital_contacts = np.random.shuffle(copy.deepcopy(self.health_workers))
        
        for patient in self.patients:
            if patient.address in hospital_contacts:
                np.delete(hospital_contacts,patient.address)

        if hospital_contacts.size >= num_assigned:
            hospital_contacts=hospital_contacts[:num_assigned]

        #otherwise take all the remaining health workers. 
        
        for edge in assigned_health_workers:
            if edge[0]>edge[1]:
                tmp = edge[0]
                edge[0] = edge[1]
                edge[1] = tmp
                
        return 
        
    def discharge_and_admit_patients(self, statuses, population_network):
        """
        (public method)
        Contact simulation is always performed on the world network,
        """
        #First modify the current population network
        self.discharge_patients(statuses, population_network)
        self.admit_patients(statuses, population_network)

        print("current patients after hospitalizations", [patient.address for patient in self.patients])
        
    def discharge_patients(self, statuses, population_network):
        '''
        (private method)
        removes a patient from self.patients and reconnect the patient with their neighbours
        if their status is no longer H
        '''
        for i, patient in enumerate(self.patients):
            if statuses[patient.address] != 'H' :
                #discharge_patient 
                population_network.remove_edges_from(patient.assigned_health_workers)
                population_network.add_edges_from(patient.community_contacts)
                print("Discharging patient", patient.address,
                      "with status", statuses[patient.address])
                del self.patients[i]
                
    def admit_patients(self, statuses, population_network):
        '''
        (private method)
        Method to find unoccupied beds, and admit patients from the populace (storing their details)
        '''
        if len(self.patients) < self.patient_capacity:
            # find unoccupied beds
            current_patients = [patient.address for patient in self.patients]
            
            populace = copy.deepcopy(self.populace)
            populace = np.delete(populace,current_patients)
            
            hospital_seeking=[i for i in populace if statuses[i] == 'H']
            print("people seeking hospitalization", hospital_seeking)

            while len(self.patients) < self.patient_capacity :
                # obtain the patients seeking to be hospitalized (state = 'H') 
                if not isinstance(hospital_seeking, list): # if it's a scalar, not array
                    hospital_seeking=[hospital_seeking]
                    
                if (len(hospital_seeking) > 0):                    
                    new_patient_address = hospital_seeking[0]
                    # create new edge between patient and corresponding health_workers
                    # Assume patients cant transmit to each other as all have disease...
                    assigned_health_workers = self.assign_health_workers(new_patient_address)
                    # store patient details
                    new_patient = Patient(new_patient_address,
                                          copy.deepcopy(list(self.static_population_network.edges(new_patient_address))),
                                          assigned_health_workers)
                                                        
                    self.patients.append(new_patient)

                    # admit patient 
                    population_network.remove_edges_from(new_patient.community_contacts)
                    population_network.add_edges_from(new_patient.assigned_health_workers)
                    print("Admitting patient from", new_patient.address,
                          "with assigned health workers", assigned_health_workers)

                    # remove hospital seeker
                    hospital_seeking.pop(0)
                    
            
            # if we reach capacity and have hospital seekers remaining
            # set hospital seekers back to i
            if (len(hospital_seeking) > 0):
                self.waiting_list.append(hospital_seeking)
                statuses[hospital_seeking] = 'I'
                print("Those placed on a waiting list as unable to get a bed ", hospital_seeking)

class Patient:
    """
    Container for current patients in hospital
    """
    def __init_(self, address, community_contacts, assigned_health_workers):
        """
        Args
        ----
        address (int): This gives location of the patient in the `static_population_network`
                      (with respect to `static_population_network.node`)

        community_contacts (list of tuples): list of edges to neighbours in `static_population_network`
                                             these are stored here while the patient is in hospital

        assigned_health_workers (list of tuples): a list of edges 
        
        """
        self.address = address
        self.community_contacts = community_contacts
        self.assigned_health_workers = assigned_health_workers
    
            
