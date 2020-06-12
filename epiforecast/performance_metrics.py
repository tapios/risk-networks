import numpy as np
import sklearn.metrics as skm

def confusion_matrix(data,
                     ensemble_states,
                     statuses  = ['I'],
                     threshold = .5,
                     reduced_system = True):

    status_catalog = dict(zip(['S', 'E', 'I', 'H', 'R', 'D'], np.arange(6)))

    ensemble_size = len(ensemble_states)
    n_status      = 5 if reduced_system else 6
    population    = len(data)

    data_statuses = [status_catalog[status] for status in list(data.values())]

    ensemble_probabilities = np.zeros((6, population))
    if reduced_system:
        ensemble_probabilities[1] = ((1 - ensemble_states.reshape(ensemble_size, 5, -1).sum(axis = 1)) > .50).mean(axis = 0)
        ensemble_probabilities[np.hstack([0,np.arange(2,6)])] = (ensemble_states.reshape(ensemble_size, n_status, population) > 0.50).mean(axis = 0)

    ensemble_statuses = ensemble_probabilities.argmax(axis = 0)

    return skm.confusion_matrix(data_statuses, ensemble_statuses, labels = np.arange(6))
