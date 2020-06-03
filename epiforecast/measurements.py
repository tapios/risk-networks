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

    def _set_prevalence(self, ensemble_states, status = 'I', reduced_system = True):
        """
        Inputs:
        -------
            ensemble_states : `np.array` of shape (ensemble_size, num_status * population) at a given time
            status_idx      : status id of interest. Following the ordering of the reduced system SIRHD.
        """
        if reduced_system == True:
            status_catalog = dict(zip(['S', 'I', 'H', 'R', 'D'], np.arange(5)))
        else :
            status_catalog = dict(zip(['S', 'E', 'I', 'H', 'R', 'D'], np.arange(6)))

        self.status = status

        n_status        = len(status_catalog.keys())
        population      = ensemble_states.shape[1]/n_status
        ensemble_size   = ensemble_states.shape[0]
        self.prevalence = ensemble_states.reshape(ensemble_size,n_status,-1)[:,status_catalog[self.status],:].sum(axis = 1)/population

    def _set_ppv(self, scale = 'log'):
        PPV = self.sensitivity * self.prevalence / \
             (self.sensitivity * self.prevalence + (1 - self.specificity) * (1 - self.prevalence))

        FOR = (1 - self.sensitivity) * self.prevalence / \
             ((1 - self.sensitivity) * self.prevalence + self.specificity * (1 - self.prevalence))

        if scale == 'log':
            logit_ppv  = np.log(PPV/(1 - PPV))
            logit_for  = np.log(FOR/(1 - FOR))

            self.logit_ppv_mean = logit_ppv.mean()
            self.logit_ppv_var  = logit_ppv.var()

            self.logit_for_mean = logit_for.mean()
            self.logit_for_var  = logit_for.var()

        else:
            self.ppv_mean = PPV.mean()
            self.ppv_var  = PPV.var()

            self.for_mean = FOR.mean()
            self.for_var  = FOR.var()

    def update_prevalence(self, ensemble_states, scale = 'log', status = 'I', reduced_system = True):
        self._set_prevalence(ensemble_states, status = status, reduced_system = reduced_system)
        self._set_ppv(scale = scale)

    def take_measurements(self, nodes_state_dict, scale = 'log', status = 'I'):
        """
        Inputs:
        -------

        """
        if self.status != status:
            print("Warning! Test is calibrated for %s, you requested %s."%(self.status, status))
            return None, None

        measurements = np.zeros(len(nodes_state_dict.keys()))
        uncertainty  = np.zeros_like(measurements)

        for node in nodes_state_dict.keys():
            if nodes_state_dict[node] == status:
                if scale == 'log':
                    measurements[node] = self.logit_ppv_mean
                    uncertainty[node]  = self.logit_ppv_var
                else:
                    measurements[node] = self.ppv_mean
                    uncertainty[node]  = self.ppv_var
            else:
                if scale == 'log':
                    measurements[node] = self.logit_for_mean
                    uncertainty[node]  = self.logit_for_var
                else:
                    measurements[node] = self.for_mean
                    uncertainty[node]  = self.for_var

        return measurements, uncertainty
