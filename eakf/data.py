
#Containers for data, includes a method to make a measurement

import numpy as np
import pickle

class HighSchoolData:

    def __init__(self,initial_steps):
        # Load the true data (beta=0.04, T=100, dt=1)
        fin = 'data/states_truth_beta_0p04.pkl'
        data = pickle.load(open(fin, 'rb'))
        data = data.T
        #we ignore the data over the initialization stage
        #NB you need dt=1.0 to do this!!
        self.data = data[initial_steps:,:]
        
    def make_observation(self,times,states):
        #noise is now added by observation class. This just obtains the data
        truth=self.data[times,:]
        truth=truth[states]
      
        return truth
