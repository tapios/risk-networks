import numpy as np

# Parent class -----------------------------------------------------------------
class Measurements:
    def __init__(self):
        pass

    def take_measurement(self, nodes_state_dict):
        pass


# Children classes -------------------------------------------------------------
class TestMeasurements(Measurements):
    def __init__(self, sensitivity = 0.80, specificity = 0.99):
        self.sensitivity = sensitivity
        self.specificity = specificity

    def set_prevalence(self, ensemble_states, status = 'I', reduced_system = True):
        """
        Inputs:
        -------
            ensemble_states : `np.array` of shape (ensemble_size, num_status * population) at a given time
            status_idx      : status id of interest. Following the ordering of the reduced system SIRHD.
        """
        if reduced_system = True
            status_catalog = dict(zip(['S', 'I', 'H', 'R', 'D'], np.arange(5)))
        else :
            status_catalog = dict(zip(['S', 'E', 'I', 'H', 'R', 'D'], np.arange(6)))

               n_status = len(status_catalog.keys())
             population = ensemble_states.shape[1]/n_status
          ensemble_size = ensemble_states.shape[0]
        self.prevalence = ensemble_states.reshape(ensemble_size,n_status,-1)[:,status_catalog[status],:].sum(axis = 1)/population

    def set_ppv(self, scale = 'log'):
        ppv = self.sensitivity * self.prevalence / \
                  (self.sensitivity * self.prevalence + (1 - self.specificity) * (1 - self.prevalence))
        if scale == 'log':
            logit_ppv  = np.log(ppv/(1 - ppv))
            self.logit_ppv_mean = logit_ppv.mean()
            self.logit_ppv_var  = logit_ppv.var()
        else:
            self.ppv_mean = ppv.mean()
            self.ppv_var  = ppv.var()

    def update_prevalence(self, ensemble_states, scale = 'log', status = 'I', reduced_system = True):
        self.set_prevalence(ensemble_states, status = status, reduced_system = reduced_system)
        self.set_ppv(scale = scale)

    def take_measurements(self, nodes_state_dict, scale = 'log', status = 'I'):
        """
        Inputs:
        -------

        """
        measurements = np.zeros(len(nodes_state_dict.nodes))
        uncertainty  = np.zeros_like(measurements)

        for node in nodes_state_dict.keys():
            if nodes_state_dict[node] == status:
                if scale == 'log':
                    measurements[node] = self.logit_ppv_mean
                    uncertainty[node]  = self.logit_ppv_var
                else:
                    measurements[node] = self.ppv_mean
                    uncertainty[node]  = self.ppv_var
