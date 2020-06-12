import numpy as np
import sklearn.metrics as skm

def confusion_matrix(data,
                     ensemble_states,
                     statuses  = ['S', 'E', 'I', 'H', 'R', 'D'],
                     threshold = 0.5,
                     combined = False):

    status_catalog = dict(zip(['S', 'E', 'I', 'H', 'R', 'D'], np.arange(6)))
    status_of_interest = [status_catalog[status] for status in statuses]
    ensemble_size = len(ensemble_states)
    population    = len(data)

    if ensemble_states.shape[1] / population == 5:
        reduced_system = True
    else:
        reduced_system = False

    n_status = 5 if reduced_system else 6

    ensemble_probabilities = np.zeros((6, population))
    if reduced_system:
        ensemble_probabilities[1] = ((1 - ensemble_states.reshape(ensemble_size, 5, -1).sum(axis = 1)) > threshold).mean(axis = 0)
        ensemble_probabilities[np.hstack([0,np.arange(2,6)])] = (ensemble_states.reshape(ensemble_size, n_status, population) > threshold).mean(axis = 0)
    else:
        ensemble_probabilities = (ensemble_states.reshape(ensemble_size, n_status, population) > threshold).mean(axis = 0)

    ensemble_statuses = ensemble_probabilities.argmax(axis = 0)
    if not combined:
        data_statuses     = [status_catalog[status] if status_catalog[status] in status_of_interest else 7 for status in list(data.values())]
        ensemble_statuses = [node_status if node_status in status_of_interest else 7 for node_status in ensemble_statuses]
    else:
        data_statuses     = [8 if status_catalog[status] in status_of_interest else 7 for status in list(data.values())]
        ensemble_statuses = [8 if node_status in status_of_interest else 7 for node_status in ensemble_statuses]
        status_of_interest = [8]
    #
    if len(status_of_interest) < 6:
        status_of_interest.insert(0, 7)

    return skm.confusion_matrix(data_statuses, ensemble_statuses, labels = status_of_interest)


def model_accuracy(data,
                   ensemble_states,
                   statuses = ['S', 'E', 'I', 'H', 'R', 'D'],
                   threshold = 0.5,
                   combined = False
                   ):

    cm = confusion_matrix(data, ensemble_states, statuses, threshold, combined)

    return np.diag(cm).sum()/cm.sum()

def f1_score(data,
             ensemble_states,
             statuses = ['E', 'I'],
             threshold = 0.5
             ):

    cm = confusion_matrix(data, ensemble_states, statuses, threshold, combined = True)
    tn, fp, fn, tp = cm.ravel()

    return 2 * tp / (2 * tp + fp + fn)
