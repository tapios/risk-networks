import numpy as np
from eakf import EAKF

        
class DAModel:
    
    def __init__(self,parameters,states,StateObservations,TimeObservations,data):

        ##Note: My idea for now is that if you create an observation model
        #       for each type you have these will get stored in osmodel[i] or otmodel[i]
        #       in the update, these will then be evaluated in the [i] pairs to collect all data.
        #       this would also require several EAKFs to be constructed to handle to different data

        #Observation model(s) for state
        self.osmodel = StateObservations
        #Observation model times
        self.otmodel = TimeObservations

        #In this simple setting we carry data here
        self.data=data

        obs_states=states[:,self.osmodel.obs_states]
        #the data assimilation model
        self.damodel = EAKF(parameters,obs_states)

        
    def update(self,x,DA_window):

        #update observation models for the new DA window
        #this should update omodel.obs_states and tmodel.obs_time
        self.otmodel.measurement(DA_window)
        self.osmodel.measurement(DA_window)
        
        print('observation at time: ', self.otmodel.obs_time,'i.e in window ', DA_window, ' time ', self.otmodel.obs_time_in_window)
        
        #Make Observation of data <Replace data with approp. model>
        #truth=self.data[self.otmodel.obs_time,:]
        #cov=np.identity(truth.shape[0])* 0.01
        #restrict to relevant states
        #truth=truth[self.osmodel.obs_states]
        #cov=cov[np.ix_(self.osmodel.obs_states,self.osmodel.obs_states)]
        
        truth,cov=self.data.make_observation(self.otmodel.obs_time , self.osmodel.obs_states)
        
        #observe ensemble states
        x=x[:,self.otmodel.obs_time_in_window,:] #same time
        x=x[:,self.osmodel.obs_states] #same place
        
        #update the da model with the data
        self.damodel.obs(truth,cov)

        #perform da model update with x
        self.damodel.update(x)

        
        
