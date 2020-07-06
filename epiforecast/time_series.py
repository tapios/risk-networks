import warnings
import numpy as np

class EnsembleTimeSeries:
    """
    Store, add, read & process a time series composed of ensemble members

    An easy way to think about this, is a 3-tensor with:
        - 0th dimension equal to n_ensemble (size of the ensemble)
        - 1st dimension equal to n_array (dimension of the stored quantity)
        - 2nd dimension equal to n_steps (total number of time steps)

    Once created, the timeseries cannot be changed in size (in any dimension).
    """

    def __init__(
            self,
            n_ensemble,
            n_array,
            n_steps):
        """
        Constructor

        Input:
            n_ensemble (int): ensemble size
            n_array (int): dimension of the vector to store
            n_steps (int): total number of time steps
        """
        self.n_ensemble = n_ensemble
        self.n_array    = n_array
        self.n_steps    = n_steps

        self.container = np.empty( (n_ensemble, n_array, n_steps) )
        self.end = 0 # points to the past-the-end element

    def __getitem__(
            self,
            timestep):
        """
        A wrapper around get_snapshot; the same docstring applies
        """
        return self.get_snapshot(timestep)

    def get_snapshot(
            self,
            timestep):
        """
        Get the full ensemble snapshot at a specified timestep

        Input:
            timestep (int): timestep of a snapshot to return

        Output:
            snapshot (np.array): (n_ensemble, n_array) array of values
        """
        if timestep >= self.end:
            raise ValueError(
                    self.__class__.__name__
                    + ": timestep is out of bounds, cannot get_snapshot"
                    + "; timestep: "
                    + str(timestep))

        return self.container[:,:,timestep]

    def get_snapshot_mean(
            self,
            timestep):
        """
        Get the ensemble mean of a snapshot at a specified timestep

        Input:
            timestep (int): timestep of a snapshot to return

        Output:
            snapshot_mean (np.array): (n_array,) array of ensemble means
        """
        snapshot = self.get_snapshot(timestep)
        return snapshot.mean(axis=0)

    def get_mean(self):
        """
        Get the ensemble mean of the whole timeseries

        Output:
            timeseries_mean (np.array): (n_array, n_steps) array of ensemble
                                        means
        """
        if self.end < self.n_steps:
            warnings.warn(
                    self.__class__.__name__
                    + ": mean of an incomplete container requested"
                    + "; the values starting from index "
                    + str(self.end)
                    + " are meaningless")

        return self.container.mean(axis=0)

    def push_back(
            self,
            snapshot):
        """
        Push back an element of time series

        Input:
            snapshot (np.array): (n_ensemble, n_array) array of values

        Output:
            None
        """
        if self.end >= self.n_steps:
            raise ValueError(
                    self.__class__.__name__
                    + ": the container is full, cannot push_back"
                    + "; end: "
                    + str(self.end)
                    + "; n_steps: "
                    + str(self.n_steps))

        self.container[:,:,self.end] = snapshot
        self.end += 1


