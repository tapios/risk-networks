
#This File has 3 sections:
# - State observations - which portion of the state is observed when observatio is carried out
# - Time observations  - what time(s) during DA window is a data collected
# - Combined observations - this just combines the two to form the observation type.
#
# E.g if i would like to observe Infectiousness for all nodes at the first time in the window
#     I can create:
#         - "IStateObservation" + "RandomTimeObservation" types
#     if i also want to observe 5% of complete states of nodes at the beginning of each window
#     I can create:
#         - "RandomStateObservation" (with obs_nodes=0.05*N input)+ "FixedTimeObservation" (with 0 input)
#

import numpy as np

### State Observations ###

#We observe the entire system during observation
class FullStateObservation:
    
    def __init__(self,N,status):
        #number of nodes in the graph
        self.N = N
        #number of different states a node can be in
        self.status = status

        #The states for observation
        self.obs_states = np.arange(status*N)

    #updates the observation model if required (here nothing changes)            
    def measurement(self,DAstep):    
        pass
    
    #enforces statuses sum to 1 if required (here nothing changes)
    def sum_to_one(self,x):
        pass
#We observe a random subset of nodes during observation
class RandomStateObservation:

    def __init__(self,N,status,obs_nodes):
        #number of nodes in the graph
        self.N = N
        #number of different states a node can be in
        self.status = status

        #The states for observation
        self.obs_nodes = obs_nodes
        #default init observation
        self.obs_states = np.arange(obs_nodes*status)

    #updates observation model (here generates a random set of nodes)
    def measurement(self,DAstep):
        #This if statement gives consistency of Random and Full State
        #otherwise the pseudorandom generator is used one extra time here
        if self.obs_nodes<self.N:
            onetoN=np.arange(self.N)
            np.random.shuffle(onetoN)#(in-place shuffle)
            tmp=np.array(sorted(onetoN[:self.obs_nodes]))
            self.obs_states=np.hstack([np.arange(self.status)+i*self.status for i in tmp])
            print(self.obs_states.shape)

    #enforces statuses sum to 1 if required (here nothing changes)        
    def sum_to_one(self,x):
        pass

class RandomStatusStateObservation:
    def __init__(self,N,status,obs_nodes,obs_status_idx):
        #number of nodes in the graph
        self.N = N
        #number of different states a node can be in
        self.status = status
        #array of status to observe
        self.obs_status_idx=obs_status_idx
        #The states for observation
        self.obs_nodes = obs_nodes
        #default init observation
        self.obs_states = np.arange(obs_nodes*obs_status_idx.size)

    #updates observation model (here generates random set of nodes)
    def measurement(self,DAstep):
        #This if statement gives consistency of Random and Full State
        #otherwise the pseudorandom generator is used one extra time here
        if self.obs_nodes<self.N:
            onetoN=np.arange(self.N)
            np.random.shuffle(onetoN)#(in-place shuffle)
            tmp=np.array(sorted(onetoN[:self.obs_nodes]))
            self.obs_states=np.hstack([self.obs_status_idx+i*self.status for i in tmp])
        else:
            onetoN=np.arange(self.N)
            self.obs_states=np.hstack([self.obs_status_idx+i*self.status for i in onetoN])

    #as we measure a subset of states, we may need to enforce other states to sum to one
    def sum_to_one(self,x):

        #CUMBERSOME... (10,6,N), sum over 2nd axis ignoring S
        #First enforce probabilities == 1, by placing excess in susceptible
        sumx=x.reshape(x.shape[0],self.status,self.N)
        sumx=np.sum(sumx[:,1:,:],axis=1)
        x[:,0:self.N]=1.0 - sumx

        #put any overflow (where sumx>1) from susceptible into exposed...
        xneg=np.minimum(x[:,0:self.N],0.0)
        x[:,0:self.N]=x[:,0:self.N]-xneg
        x[:,self.N:2*self.N]=x[:,self.N:2*self.N]-xneg
        
        #print("total mass put from susceptible => exposed: ", np.sum(np.sum(xneg,axis=1),axis=0))
        return x
        
            
### --- Time Observations --- ###

            
#We observe at a fixed time in each DA window
class FixedTimeObservation:

    def __init__(self,t_range,obs_idx):

        #The timesteps of a DA window
        self.t_range = t_range #[dt,2dt,....,T-dt,T]
        self.Tsize = t_range.size 

        #storage for time overall, and local time wrt window
        #note the index corresponds to the index in t_range
        self.obs_time_in_window=obs_idx
        self.obs_time= self.obs_time_in_window
        
    def measurement(self,DAstep):
        #time_in_window is fixed
        self.obs_time = DAstep*self.Tsize  + self.obs_time_in_window

### --- We observe at a random time in each DA window --- ###       
class RandomTimeObservation:

    def __init__(self,t_range):

        #The timesteps of a DA window
        self.t_range = t_range #[dt,2dt,....,T-dt,T]
        self.Tsize = t_range.size 
        
        #storage for time overall, and local time wrt window
        self.obs_time_in_window=np.random.randint(self.Tsize)
        self.obs_time=self.obs_time_in_window
        
    def measurement(self,DAstep):
        #randomly generate new time
        self.obs_time_in_window=np.random.randint(self.Tsize)
        self.obs_time = DAstep*self.Tsize +self.obs_time_in_window


### --- Combined Observation --- ###

#Random Observation 
class RandomObservation(RandomTimeObservation,RandomStateObservation):

    def __init__(self,t_range,N,status,obs_nodes,obs_name):
    
        RandomTimeObservation.__init__(self,t_range)
        RandomStateObservation.__init__(self,N,status,obs_nodes)
        self.name=obs_name

    def measurement(self,DAstep):
        RandomTimeObservation.measurement(self,DAstep)
        RandomStateObservation.measurement(self,DAstep)

#Random Observation of a defined set of status's SEIHRD (by idx)
class RandomStatusObservation(RandomTimeObservation,RandomStatusStateObservation):

    def __init__(self,t_range,N,status,obs_nodes,status_obs_idx,obs_name):

        if not isinstance(status_obs_idx,np.ndarray):#if it's a scalar, not array
            status_obs_idx=np.array(status_obs_idx)

        RandomTimeObservation.__init__(self,t_range)
        RandomStatusStateObservation.__init__(self,N,status,obs_nodes,status_obs_idx)
        self.name=obs_name
        
    def measurement(self,DAstep):
        RandomTimeObservation.measurement(self,DAstep)
        RandomStatusStateObservation.measurement(self,DAstep)
