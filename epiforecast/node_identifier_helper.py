def load_node_identifiers(self, filename=None):
   if filename is None:
     filename = self.fallback_node_identifier_filename

   node_identifiers = np.loadtxt(filename, dtype = str)
   nodes = data[:,0].astype(np.int)
   identifiers = data[:,1]
   
   # number of nodes without hospital beds
   pop = len(identifiers[identifiers != 'HOSP'])
   
   # something has to be done with these identifiers later on in setIC

