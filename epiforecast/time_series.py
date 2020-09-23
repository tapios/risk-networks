import warnings
import numpy as np

class EnsembleTimeSeries:
    """
    Store, add, read & process a time series composed of ensemble members

    An easy way to think about this, is a 3-tensor with:
        - 0th dimension equal to n_ensemble (size of the ensemble)
        - 1st dimension equal to size_state (dimension of the stored quantity)
        - 2nd dimension equal to n_steps (total number of time steps)

    Once created, the timeseries cannot be changed in size (in any dimension).
    """

    def __init__(
            self,
            n_ensemble,
            size_state,
            n_steps,
            update_batch=1):
        """
        Constructor

        Input:
            n_ensemble (int): ensemble size
            size_state (int): dimension of the vector to store
            n_steps (int): (minimum) number of time steps to store
                           total number of stored steps will be n_steps + update_batch   
            update_batch (int): perform a shift every update_batch timesteps 
        """
        assert n_steps > update_batch

        self.n_ensemble = n_ensemble
        self.size_state    = size_state
        self.n_steps    = n_steps

        self.container = np.empty( (n_ensemble, size_state, n_steps) )
        self.update_batch = update_batch
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
            snapshot (np.array): (n_ensemble, size_state) array of values
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
            snapshot_mean (np.array): (size_state,) array of ensemble means
        """
        snapshot = self.get_snapshot(timestep)
        return snapshot.mean(axis=0)

    def get_mean(self):
        """
        Get the ensemble mean of the whole timeseries

        Output:
            timeseries_mean (np.array): (size_state, n_steps) array of ensemble
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
        Push back an element of time series. If it is full,
        we shift the data to overwrite the first `self.update_batch` elements

        Input:
            snapshot (np.array): (n_ensemble, size_state) array of values

        Output:
            None
        """

        # if self.end >= self.n_steps:
        #     raise ValueError(
        #             self.__class__.__name__
        #             + ": the container is full, cannot push_back"
        #             + "; end: "
        #             + str(self.end)
        #             + "; n_steps: "
        #             + str(self.n_steps))
        
        # if we are at capacity, lose the first entry
        if self.end >= self.n_steps:
            self.container = np.roll(self.container, -self.update_batch, axis=2)
            self.end -= self.update_batch
            self.container[:,:,self.end] = snapshot
            self.end += 1
         
        else:
            self.container[:,:,self.end] = snapshot
            self.end += 1
    

        

