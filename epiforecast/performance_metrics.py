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


class ModelAccuracy:
    """
    Container for model accuracy metric. Metric based on overall class assignment.
                Accuracy = TruePositives + TrueNegatives / TotalCases
    """

    def __init__(self, name = 'Accuracy'):
        self.name = name

    def __call__(self,
                 data,
                 ensemble_states,
                 statuses = ['S', 'E', 'I', 'H', 'R', 'D'],
                 threshold = 0.5,
                 combined = False
                 ):
        """
            Args:
            -----
                data           : dictionary with {node : status}
                ensemble_state : (ensemble size, 5 * population) `np.array` with probabilities
                statuses       : statuses of interest.
                threshold      : used to declare a positive class.
                combined       : if statuses in `statuses` are to be taken as a single class.
        """
        cm = confusion_matrix(data, ensemble_states, statuses, threshold, combined)
        return np.diag(cm).sum()/cm.sum()

class F1Score:
    """
    Container for the F1 score metric. Score used for highly unbalanced data sets.
    Harmonic mean of precision and recall.
            F1 = 2 / ( recall^-1 + precision^-1).
    """

    def __init__(self, name = 'F1 Score'):
        self.name = name

    def __call__(self,
                 data,
                 ensemble_states,
                 statuses = ['E', 'I'],
                 threshold = 0.5
                 ):
        """
        Glossary:
            tn : true negative
            fp : false positive
            fn : false negative
            tp : tru positive
        """
        cm = confusion_matrix(data, ensemble_states, statuses, threshold, combined = True)
        tn, fp, fn, tp = cm.ravel()

        return 2 * tp / (2 * tp + fp + fn)

class PerformanceTracker:
    """
    Container to track how a classification model behaves over time.
    """
    def __init__(self,
                  metrics   = [ModelAccuracy(), F1Score()],
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

    def __str__(self):
        """
            Prints current metrics.
        """
        print(" ")
        print("=="*30)
        for kk, metric in enumerate(self.metrics):
            print("[ %s ]                          : %.4f,"%(metric.name, self.performance_track[-1,kk]))
        print("=="*30)
        return ""

    def eval_metrics(self,
                     data,
                     ensemble_states):
        """
        Evaluates each metric in list of metrics.
        Args:
        -----
            data: dictionary with {node : status}
            ensemble_state: (ensemble size, 5 * population) `np.array` with probabilities
        """

        results = [metric(data, ensemble_states, self.statuses, self.threshold) for metric in self.metrics]
        if self.performance_track is None:
            self.performance_track = np.array(results).reshape(1, len(self.metrics))
        else:
            self.performance_track = np.vstack([self.performance_track, results])

    def eval_prevalence(self, data):
        """
        Evaluates the prevalence of the status of interest in `self.statuses`.
        If multiple, it combines them as a single status.
            Args:
            -----
                data: dictionary with {node : status}
        """

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
        """
        Evaluates both the prevalence of the status of interest in `self.statuses`,
        and the performance metrics given a snapshot of the current state of the
        sytem (`kinetic model`) and the model (ensemble 'master equations').

            Args:
            -----
                data: dictionary with {node : status}
                ensemble_state: (ensemble size, 5 * population) `np.array` with probabilities
        """

        self.eval_metrics(data, ensemble_states)
        self.eval_prevalence(data)
