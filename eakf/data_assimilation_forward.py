import numpy as np
from eakf_multiobs import EAKF

#This class holds the data model, the observation model(s), the DA for each observation type
#the update method applies Per Window.
#it will find all data from window, put them in forwards chronological order, and filter them:
#i.e DA on first point, update ensemble, run to 2nd point, DA on 2nd point, etc.
class DAForwardModel:

    def __init__(self,parameters,states,Observations,data):

        #observation models(s)
        self.omodel = Observations
        
        #In this simple setting we carry data here
        self.data=data

        self.params=parameters[np.newaxis]
        
        obs_states=[states[:,self.omodel[i].obs_states] for i in range(len(self.omodel))]
        #the data assimilation models (One for each observation model)
        self.damodel = [EAKF(obs_states[i]) for i in range(len(self.omodel))]

        self.data_pts_assimilated=0

    def order_obs_times_states(self,DA_window):

        for i in np.arange(len(self.omodel)):
            #update observation models for the new DA window
            #this should update omodel.obs_states and omodel.obs_time
            self.omodel[i].measurement(DA_window)
            
            
        #Order the models chronologically (Still assume just 1 point from each model atm)
        #first zip models together, sort by obs_time, then 'transpose and unzip', then convert from resulting tuples to lists
        self.omodel,self.damodel = (list(t) for t in zip(*sorted(zip(self.omodel,self.damodel), key=lambda x: x[0].obs_time)))

        self.data_pts_assimilated=0
      
        print(len(self.omodel), 'data points to be assimilated in window', DA_window )
        for i in range(len(self.omodel)):
            print('observation ', self.omodel[i].name ,'at time: ', self.omodel[i].obs_time)#,', i.e in window ', DA_window, ' time ', self.omodel[i].obs_time_in_window)

            
    def update(self,x):

        #Call order_obs_times_states before this to obtain the observation times with the models, with self.omodel re-ordered chronologically in the window

        pt=self.data_pts_assimilated
        om=self.omodel[pt]
        dam=self.damodel[pt]
        
        #naive 1st attempt - update one point at a time
        truth,cov=self.data.make_observation(om.obs_time , om.obs_states)
                       
        #observe ensemble states
        x=x[:,om.obs_time_in_window,:] #same time
        x=x[:,om.obs_states] #same place
                       
        #update the da model with the data
        dam.obs(truth,cov)

        #perform da model update with x,q and update parameters.
        q=dam.update(x,self.params[-1])
        np.append(self.params, [q], axis=0)

        self.data_pts_assimilated = pt+1
        

    #the next observation time if we have one, (it could be the same)
    #else if we have run out of points, give the end of window value
    def next_data_idx(self):
        next_pt=self.data_pts_assimilated
        pt=next_pt-1
        
        if next_pt<len(self.omodel):
            return self.omodel[next_pt].obs_time_in_window
        else:
            return self.omodel[0].Tsize-1
        
