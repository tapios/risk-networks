import numpy as np
from eakf import EAKF

        
class DAModel:
    
    def __init__(self,parameters,states,Observations,data):

        ##Note: My idea for now is that if you create an observation model
        #       for each type you have these will get stored in osmodel[i] or otmodel[i]
        #       in the update, these will then be evaluated in the [i] pairs to collect all data.
        #       this would also require several EAKFs to be constructed to handle to different data

        #Observation model(s) for state
        #self.osmodel = StateObservations
        #Observation model times
        #self.otmodel = TimeObservations

        #observation models(s)
        self.omodel = Observations
        
        #In this simple setting we carry data here
        self.data=data

        obs_states=states[:,self.omodel.obs_states]
        #the data assimilation model
        self.damodel = EAKF(parameters,obs_states)

        
    def update(self,x,DA_window):

        #update observation models for the new DA window
        #this should update omodel.obs_states and omodel.obs_time
        self.omodel.measurement(DA_window)       
        print('observation at time: ', self.omodel.obs_time,'i.e in window ', DA_window, ' time ', self.omodel.obs_time_in_window)
        truth,cov=self.data.make_observation(self.omodel.obs_time , self.omodel.obs_states)
        
        #observe ensemble states
        x=x[:,self.omodel.obs_time_in_window,:] #same time
        x=x[:,self.omodel.obs_states] #same place
        
        #update the da model with the data
        self.damodel.obs(truth,cov)

        #perform da model update with x
        self.damodel.update(x)

        
        
