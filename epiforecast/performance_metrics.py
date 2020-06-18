import numpy as np
import sklearn.metrics as skm
from collections import defaultdict

def confusion_matrix(data,
                     ensemble_states,
                     statuses  = ['S', 'E', 'I', 'H', 'R', 'D'],
                     threshold = 0.5,
                     combined = False):

    """
    Wrapper of `sklearn.metrics.confusion_matrix`.
    Args:
    -----
        ensemble_states: (ensemble_size, 5 * population) `np.array` of the current state of the ensemble ODE system.
        statuses: list of statuses of interest.
        threshold for declaring a given status.
        combined: Boolean that allows to treat a classification as one vs the rest.
    """

    status_catalog = dict(zip(['S', 'E', 'I', 'H', 'R', 'D'], np.arange(6)))
    status_of_interest = [status_catalog[status] for status in statuses]
    ensemble_size = len(ensemble_states)
    population    = len(data)

    n_status = 5

    ensemble_probabilities = np.zeros((6, population))

    ensemble_probabilities[1] = ((1 - ensemble_states.reshape(ensemble_size, 5, -1).sum(axis = 1)) > threshold).mean(axis = 0)
    ensemble_probabilities[np.hstack([0,np.arange(2,6)])] = (ensemble_states.reshape(ensemble_size, n_status, population) > threshold).mean(axis = 0)

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
    """
    Metric based on overall class assignment.
            Accuracy = TP + TN / Total cases
    """

    cm = confusion_matrix(data, ensemble_states, statuses, threshold, combined)

    return np.diag(cm).sum()/cm.sum()

def f1_score(data,
             ensemble_states,
             statuses = ['E', 'I'],
             threshold = 0.5
             ):
    """
    Score used for highly unbalanced data sets. Harmonic mean of precision and recall.
            F1 = 2 / ( recall^-1 + precision^-1)
    """
    cm = confusion_matrix(data, ensemble_states, statuses, threshold, combined = True)
    tn, fp, fn, tp = cm.ravel()

    return 2 * tp / (2 * tp + fp + fn)

class PerformanceTracker:
    """
    Container to track how a classification model behaves over time.
    """
    def __init__(self,
                  metrics   = [model_accuracy, f1_score],
                  statuses  = ['E', 'I'],
                  threshold = 0.5):
        """
        Args:
        ------
            metrics: list of metrics that can be fed to the wrapper.
            statuses: statuses of interest.
            threhold: 0.5 by default to declare a given status.
        """

        self.statuses  = statuses
        self.metrics   = metrics
        self.threshold = threshold
        self.performance_track = None
        self.prevalence_track  = None

    def print(self):
        print(" ")
        print("=="*30)
        print("[ Accuracy ]                          : {:.4f},".format(self.performance_track[-1,0]))
        print("[ F1 Score ]                          : {:.4f},".format(self.performance_track[-1,1]))
        print("=="*30)

    def eval_metrics(self,
                     data,
                     ensemble_states):

        results = [metric(data, ensemble_states, self.statuses, self.threshold) for metric in self.metrics]
        if self.performance_track is None:
            self.performance_track = np.array(results).reshape(1, len(self.metrics))
        else:
            self.performance_track = np.vstack([self.performance_track, results])

    def eval_prevalence(self, data):

        status_catalog = dict(zip(['S', 'I', 'H', 'R', 'D'], np.arange(5)))
        population = len(data)

        a, b = np.unique([v for v in data.values()], return_counts=True)
        status_counts = defaultdict(int, zip(a, b))

        prevalence = np.array([status_counts[status] for status in self.statuses]).sum()/population

        if self.prevalence_track is None:
            self.prevalence_track = np.array(prevalence)
        else:
            self.prevalence_track = np.hstack([self.prevalence_track, prevalence])

    def update(self, data, ensemble_states):

        self.eval_metrics(data, ensemble_states)
        self.eval_prevalence(data)