
import numpy as np

class HealthService:
    """
    Class to represent the actions of a health service for the population,
    Contains 1 core method, `discharge_and_obtain_patients(node_statuses) that swaps nodes
    of status 'H' (hospitalized) with 'P' inert placeholders that represent hospital beds.
    When 'H' is in hospital, it has a new contact network of only health workers, and the
    inert 'P' is replaced in the community (or health workers) as acts as space in the graph
    """
    
    def __init__(self,node_identifiers):
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
    
    def discharge_and_obtain_patients(self,node_statuses):
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

    def __vacate_hospital_beds(self,node_statuses):
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

        
    def __populate_hospital_beds(self,node_statuses):
        '''
        (private method)
        Put 'H' nodes currently outside hospital beds (`hospital_seeking`), into an unoccupied hospital bed (`new_patient`).
        Record where the patient came from (in `self.patient_home`) place a 'P' in it's network position.
        '''
        #check each hospital bed
        for hospital_bed in self.node_identifiers["hospital_beds"]:
            #if any bed is unoccupied, then value == 'P'
            if node_statuses[hospital_bed] == 'P':
                
                #obtain the nodes seeking to be hospitalized (state == 'H') 
                populace = np.hstack([self.node_identifiers["health_workers"] , self.node_identifiers["community"]])
  
                hospital_seeking=[i for i in populace if node_statuses[i]== 'H']
                if (len(hospital_seeking)>0):
                    new_patient_address = hospital_seeking[0]      
                    #move a patient into the hospital bed, keeping track of its address
                    node_statuses[hospital_bed] = 'H'
                    node_statuses[new_patient_address] = 'P'
                    self.address_of_patient[hospital_bed]= new_patient_address
                    print("receiving new patient from",new_patient_address," into bed ",hospital_bed)
       


