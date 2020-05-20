import numpy as np
from eakf_multiobs import EAKF

#This class holds the data model, the observation model(s), the DA for each observation type
#the update method applies Per Window.
#it will find all data from window, put them in forwards chronological order, and filter them:
#i.e DA on first point, update ensemble, run to 2nd point, DA on 2nd point, etc.
class ForwardAssimilator:

    def __init__(self,parameters,states,Observations,Errors):

        if not isinstance(Observations,list):#if it's a scalar, not array
            Observations=[Observations]

        if not isinstance(Errors,list):#if it's a scalar, not array
            Errors=[Errors]
       
        #observation models(s)
        self.omodel = Observations      

        #store the parameters
        self.params=parameters[np.newaxis]
        
        obs_states=[states[:,self.omodel[i].obs_states] for i in range(len(self.omodel))]
        #the data assimilation models (One for each observation model)
        self.damethod = EAKF(obs_states[0])

        self.data_pts_assimilated=0
        #online evaluations of errors, one needs an observation class to check differences in data
        self.online_emodel= Errors
        
    def initialize_obs_in_window(self,DA_window):

        for i in np.arange(len(self.omodel)):
            #update observation models for the new DA window
            #this should update omodel.obs_states and omodel.obs_time
            self.omodel[i].initialize_new_window(DA_window)
            
            
        #Order the models chronologically (Still assume just 1 point from each model atm)
        self.omodel =sorted(self.omodel, key=lambda x: x.obs_time)

        self.data_pts_assimilated=0
      
        print(len(self.omodel), 'data points to be assimilated in window', DA_window )
        for i in range(len(self.omodel)):
            print('observation ', self.omodel[i].name ,'at time: ', self.omodel[i].obs_time)

            
    def update(self,x,data):

        pt=self.data_pts_assimilated
        om=self.omodel[pt]
        dam=self.damethod
      
        #Restrict x to the the observation time
        x=x[:,om.obs_time_in_window,:]
          
        om.make_new_obs(x) #Generate states to observe at observation time 
      
        if (om.obs_states.size>0):

            print("partial states to be assimilated", om.obs_states.size)

            truth = data.make_observation(om.obs_time,om.obs_states)
            cov = om.get_observational_cov(truth)
            
            #perform da model update with x,q and update parameters.
            q,x[:,om.obs_states]=dam.update(x[:,om.obs_states],self.params[-1],truth,cov)
            np.append(self.params, [q], axis=0)
         
            #Force probabilities to sum to one    
            om.sum_to_one(x)
            
            print("EAKF error:", dam.error[-1])
        else:
            print("no assimilation required")

        #Error to truth
        self.error_to_truth_state(x,data)

        #iterate to next point
        self.data_pts_assimilated = pt+1

        
        #return x at the observation time
        return self.params[-1],x,self.next_data_idx()
        
    #the next observation time if we have one, (it could be the same)
    #else if we have run out of points, give the end of window value
    def next_data_idx(self):
        next_pt=self.data_pts_assimilated
        pt=next_pt-1
        
        if next_pt<len(self.omodel):
            return self.omodel[next_pt].obs_time_in_window
        else:
            return self.omodel[0].Tsize-1
        
    #defines a method to take a difference to the data state
    def error_to_truth_state(self,state,data):
        
        em=self.online_emodel[self.data_pts_assimilated] #get corresponding error model
        otime=self.omodel[self.data_pts_assimilated].obs_time
        #Make sure you have a deterministic ERROR model - or it will not match truth
        #without further seeding
        em.make_new_obs(state)
        predicted_infected = em.obs_states
        truth=data.make_observation(otime,np.arange(em.status*em.N))
        print(truth[predicted_infected])
        em.make_new_obs(truth[np.newaxis,:])
        
        #take error
        actual_infected= em.obs_states
        print(truth[actual_infected])
        different_states=np.zeros(em.status*em.N)
        different_states[predicted_infected]=1.0
        different_states[actual_infected] = different_states[actual_infected]-1.0
        number_different=np.maximum(different_states,0.0)-np.minimum(different_states,0.0)
        print(np.sum(number_different))
