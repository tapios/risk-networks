import numpy as np
def load_node_identifiers(filename):
 
   data = np.loadtxt(filename, dtype = str)
   
   nodes = data[:,0].astype(np.int)
   identifiers = data[:,1]

   hospital_beds  = nodes[identifiers == 'HOSP']
   health_workers = nodes[identifiers == 'HCW']
   community      = nodes[identifiers == 'CITY']
  
   return hospital_beds, health_workers, community

