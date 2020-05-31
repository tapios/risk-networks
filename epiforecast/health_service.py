import copy
import numpy as np

class HealthService:
    """
    Class to represent the actions of a health service for the population,
    This class is initiated with a full static network of hospital_beds,
    health_workers and community known as the `world_network`. The important
    structure of this network are the hospital_beds, that are nodes linked
    to the community network, only through a layer of health workers:

    |-----------------------World network------------------------|

    [hospital_bed 1]---[health_worker 1: 'S']---[community 1: 'S'] 
                    \ /                      \ /    
                     X                        X-[community 2: 'I'] 
                    / \                      / \    
    [hospital_bed 2]---[health_worker 2: 'S']---[community 3: 'H']

    |------------------------------------------------------------|

    The primary method of this class is:

    population_network = `discharge_and_obtain_patients(current_statuses)`

    The output is a network identified with only human beings, known as the `population network`
    (no hospital beds) and the connectivity is different from world network as given by the
    following:

    (obtain) Any `community` or `health_worker` node that (and not already a `patient`)
    with status 'H' (hospitalized)  will lose their edges connecting them to current
    neighbours and gain the edges of an unoccupied hospital. We refer to them as a `patient`.

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

    def __init__(self, node_identifiers, world_network):
        """
        Args
        ----
        world_network (networkx graph): This is the full (static) network of possible contacts for
                                        `hospital_beds`, `health_workers` and `community` nodes

        node_identifiers (dict [np.array]): a dict of size 3;
                          ["hospital_beds"] contains the node indices of the hospital beds
                          ["health_workers"] contains the node indices of the health workers
                          ["community"] contains the node indices of the community
                                       
        """
        self.world_network=world_network
        
        self.node_identifiers = node_identifiers
        self.patient_capacity = node_identifiers["hospital_beds"].size   
        # Stores { "hospital bed" , "address" , "population_contacts"} of patient nodes in a dict 
        self.patients = []

        
    def create_population_network(self):
        """
        Creates the network for KineticModel to simulate on, includes health_workers and
        community nodes, but without hospital beds
        """
        population_network=copy.deepcopy(world_network)        
        population_network.remove_nodes_from(self.node_identifiers["hospital_beds"])
        return population_network
        
    def discharge_and_admit_patients(self, population_statuses, averaged_contact_duration, population_network):
        """
        (public method)
        Contact simulation is always performed on the world network, therefore 
        """
        #First create the current contact network
        self.__discharge_patients(population_statuses, population_network)
        self.__admit_patients(population_statuses, population_network)

        self.__set_edge_weights(averaged_contact_duration, population, population_network)     
        
    def __discharge_patients(self, node_statuses, population_network):
        '''
        (private method)
        removes a patient from self.patients and reconnect the patient with their neighbours
        if their status is no longer H
        '''
        for i,patient in enumerate(self.patients):
            if node_statuses[patient["address"]] != 'H' :
                #discharge_patient 
                nx.remove_edges_from(patient["hospital_contacts"])
                nx.add_edges_from(patient["community_contacts"])
                del self.patients[i]

    def __admit_patients(self, population_network,node_statuses):
        '''
        (private method)
        Method to find unoccupied beds, and admit patients from the populace (storing their details)
        '''
        if len(self.patients) < self.patient_capacity:
            #find unoccupied beds
            occupied_beds = [self.patients[i]["hospital_bed"] for i in range(len(self.patients))]
            unoccupied_beds=np.delete(np.arange(self.patient_capacity), occupied_beds)

            for bed in unoccipied_beds:
                #obtain the patients seeking to be hospitalized (state = 'H') 
                populace = np.hstack([self.node_identifiers["health_workers"] , self.node_identifiers["community"]])
                hospital_seeking=[i for i in populace if node_statuses[i] == 'H']
                if (len(hospital_seeking) > 0):
                    new_patient = hospital_seeking[0]
                    #create new edge for patient to corresponding hospital_workers
                    hospital_contacts=list(copy.deepcopy(self.world_network.neighbors(bed)))
                    hospital_contacts=[tuple(new_patient,i) for i in bed_contacts]
                    #store patient details
                    new_patient_details = {"address"            : new_patient,
                                           "community_contacts" : copy.deepcopy(self.world_network.edges(new_patient)),
                                           "hospital_bed"       : bed,
                                           "hospital_contacts"  : hospital_contacts}
                    
                    self.patients.append(new_patient_details)

                    # admit patients 
                    nx.remove_edges_from(new_patient_details["community_contacts"])
                    nx.add_edges_from(new_patient_details["hospital_contacts"])

                    
    def __set_edge_weights(averaged_contact_duration, population, population_network)     :

        #Get occupied beds,
        #remove patient["community_contact"] weights
        #swap occupied beds for patients (patient["bed"] , worker) = XXX -> (patient["address"], worker) = XXX
        extend weights by the new edges to patients. 
        #remove weights for hospital bed nodes
        if len(self.patients) < self.patient_capacity:
            #find unoccupied beds
            occupied_beds = [self.patients[i]["hospital_bed"] for i in range(len(self.patients))]

        
        
        #Then set weights on new network
        weights = {tuple(edge): averaged_contact_duration[i] 
                   for i, edge in enumerate(nx.edges(self.population_network))}
        
        #These are the only attributes that are modified when changing edges.
        nx.set_edge_attributes(population_network, values=weights, name='SI->E')
        nx.set_edge_attributes(population_network, values=weights, name='SH->E')
   
            
class HealthService:
    """
    Class to represent the actions of a health service for the population,
    Contains 1 core method, `discharge_and_obtain_patients(node_statuses) that swaps nodes
    of status 'H' (hospitalized) with 'P' inert placeholders that represent hospital beds.
    When 'H' is in hospital, it has a new contact network of only health workers, and the
    inert 'P' is replaced in the community (or health workers) as acts as space in the graph
    """
    
    def __init__(self, node_identifiers):
        """
        Args
        ---
        node_identifiers (dict[np.array]):
        a dict of size 3, ["hospital_beds"] contains the node indices of the hospital beds
                          ["health_workers"] contains the node indices of the health workers
                          ["community"] contains the node indices of the community
        """
        
        self.node_identifiers = node_identifiers
        #Stores the addresses of the current patients in a hospital bed [i]
        self.address_of_patient = np.repeat(-1, node_identifiers["hospital_beds"].size)
    
    def discharge_and_obtain_patients(self, node_statuses):
        """
        (public method)
        This runs the following:
        1) We vacate any hospital_bed nodes of patients no longer of status 'H'
           returning them to the original address
        2) Then, populate any vacant hospital_bed nodes with patients of status 'H'
           (if there are any), recording the address from where they came 
        This method modifies the node_statuses object
        """
        self.__vacate_hospital_beds(node_statuses)
        self.__populate_hospital_beds(node_statuses)

    def __vacate_hospital_beds(self, node_statuses):
        '''
        (private method)
        Vacate hospital_bed nodes if their status is not 'H'
        '''
        #check each hospital bed 
        for hospital_bed in self.node_identifiers["hospital_beds"]:
            #If there is a node occupying  the bed  then the value != 'P'
            #If the node is longer in hospitalized state then the value != 'H'
            if node_statuses[hospital_bed] != 'P' and node_statuses[hospital_bed] != 'H':
                #then move them back to their original nodal position
                node_statuses[self.address_of_patient[hospital_bed]] = node_statuses[hospital_bed]
                #and set the state of the bed back to unoccupied: 'P'
                print("sending home patient ", hospital_bed, " to ", self.address_of_patient[hospital_bed], " in state ", node_statuses[hospital_bed])
                node_statuses[hospital_bed] = 'P'
                self.address_of_patient[hospital_bed] = -1 #an unattainable value may be useful for debugging

        
    def __populate_hospital_beds(self, node_statuses):
        '''
        (private method)
        Put 'H' nodes currently outside hospital beds (`hospital_seeking`), into an unoccupied hospital bed (`new_patient`).
        Record where the patient came from (in `self.patient_home`) place a 'P' in it's network position.
        '''
        #check each hospital bed
        for hospital_bed in self.node_identifiers["hospital_beds"]:
            #if any bed is unoccupied, then value = 'P'
            if node_statuses[hospital_bed] == 'P':
                
                #obtain the nodes seeking to be hospitalized (state = 'H') 
                populace = np.hstack([self.node_identifiers["health_workers"] , self.node_identifiers["community"]])
  
                hospital_seeking=[i for i in populace if node_statuses[i] == 'H']
                if (len(hospital_seeking) > 0):
                    new_patient_address = hospital_seeking[0]      
                    #move a patient into the hospital bed, keeping track of its address
                    node_statuses[hospital_bed] = 'H'
                    node_statuses[new_patient_address] = 'P'
                    self.address_of_patient[hospital_bed] = new_patient_address
                    print("receiving new patient from", new_patient_address, " into bed ", hospital_bed)
       


