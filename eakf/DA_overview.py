#This comprises a list of empty classes and how they fit in an example:
#The example
###########################
#Forward_Filter_EXAMPLE.PY#
###########################
#The other classes we use
from data import dataobject
from observations import ObservationModel
from data_assimilation_forward import DAForwardModel

def load_G():
    #Loads a graph

def run_model(0,T_init,model_run_params):
    #Parallelization 
    #runs model for certain final time time T_init, outputs at given times.

    
main:

load_G()#Load graph

#Input DA steps, Ensemble size, etc.
...

#initialization for first forward model run
states_IC = ...

#Initialization Forward run (Pre DA), e.g run from 0 to T_init
run_forward_model(0,T_init,params)

#Now, for DA stuff
#Create a data object - see data.py below
data=DataObject() 
#Observation objects - see observations.py below
infectious=ObservationModel(..., status=2)#
hospital=ObservationModel(...,status=3)
omodels =[infectious,hospitalized] #Combine

#Data assimilation model
ekf=DAForwardModel(...,initial_states,omodels, data)

#DA loop
for DA_window in range(DA_total_windows):

    #first run Forward model for ensembles over whole window
    xforward=run_model(Tlast_window, T_next_window,params)

    #get the times/states of observations over the window
    ekf.order_obs_times_states(DA_window)

    #next data at data_pt_time
    data_time=ekf.next_data_idx()

    while(data pts assimilated < no. observations):
        #Assimilate ensembles with truth here
        x_forward[data_time] = ekf.update(x_forward)
        
        #get next data point
        old_data_time=data_time
        data_time=ekf.next_data_idx()

        #propagate model to next data point
        xforward[old_data_time:data_time] = run_model(old_data_time,data_time)


########################
#Now we will go through the Different classes we have in files:
#data.py
#observations.py
#eakf.py
#data_assimilation forward

#########
#DATA.PY#
#########

class DataObject:

    def __init__(self):
        #load up some data source from file, or
        #initialized data model
        
    def make_observation(self,time,state):
        #Given a time and state, observe the data
        #e.g read from stored data
        #or propagate the data model to the time/state
        return truth_mean, truth_covariance
    
########################

#################
#observations.py#
#################

class ObservationModel:

    def __init__(self, time_params, state_params):
        #initialize time and state observation models
        self.obs_times
        self.obs_states
    def initialize_new_window(self,DA_window):
        #update obs_times and obs_states for the new window
        #This could be 
        # - full/reduced statuses (S,E,I,H,R,D)
        # - at a fixed/random selection of nodes (1:N)
        # - at a fixed/random time in ([dt,...,K*dt=T])
        # Only 1 obs time per obs model per window (currently)

    def initialize_new_obs(self,state):
        #update obs_states at the obs_time, important if
        #we observe based on the current state.

        
#########################

#########
#eakf.py#
#########

class EAKF:

    def __init__(self,init_states):
        self.state = init_states
        self.error=0

    def update(self,states,model_params,truth_mean, truth_cov):
        #updates states and model_params based on the truth mean & covariance 
        return states,model_params

###########################
    
##############################
#data_assimilation_forward.py#
##############################

class DAForwardModel:

    def __init__(self,model_parameters,initial_states,observations,data):
        #store observations, data, parameters and the EAKF for each observation set
        self.omodel = observations
        self.data = data
        self.params = model_parameters
        self.damodel=[EAKF(observations[i]) for i in range(len(observations))]
                      
        def initialize_obs_in_window(self,DA_window):
            for i in range(len(self.omodel)): #for each observation type
                self.omodel[i].initialize_new_window(DA_window) #Generate the observation points

            #Then Chronologically order self.omodel, self.damodel

        def update(self,x):

            self.omodel[pt].initialize_new_obs(x)
            
            #for current data point pt:
            obs_time=self.omodel[pt].obs_times
            obs_states=self.omodel[pt].obs_states
            dam=self.damodel[pt]

            #obtain data (&give to eakf)
            truth_mean,truth_cov = data.make_observation(obs_time,obs_states)
            dam.obs(truth_mean,truth_cov)

            #update x, with corresponding EAKF
            self.params,x[obs_time,obs_states] =dam.update(x[obs_time,obs_states])

            return x
            
            
