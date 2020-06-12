import copy
from collections import namedtuple

class StaticNetwork:
    """
    A container to hold a static contact network, it contains
    The network, when it used in the simulation, and the data (statuses)
    at these times.
    """
    def __init__(self,
                 contact_network,
                 start_time,
                 end_time):
        
        self.contact_network = copy.deepcopy(contact_network)
        self.start_time = copy.deepcopy(start_time)
        self.end_time = copy.deepcopy(end_time)
   
    def set_end_statuses(self, end_statuses):
        self.end_statuses = copy.deepcopy(end_statuses)
        
    def set_start_statuses(self, start_statuses):
        self.start_statuses = copy.deepcopy(start_statuses)
   
class StaticNetworkSeries:
    """
    A container to hold a series of StaticNetwork objects. It stores the networks as a
    dictionary with keys given by a named tuple StartEndTime which will set/get networks based
    on the provided start_time, end_time or both
    """
    def __init__(self, static_contact_interval):
        """
        Args
        ----
        static_contact_interval (float): the fixed duration at which the network is static. (so we can
                                         deduce end time from start time, start_time from end time).
        """
        self.static_network_series={}
        self.static_contact_interval=static_contact_interval 
        self.StartEndTime = namedtuple("StartEndTime",["start","end"])
        
    def save_network_by_start_time(self,
                                   contact_network,
                                   start_time):

        end_time = start_time+self.static_contact_interval
        start_end_time = self.StartEndTime(start=start_time, end=end_time)
        new_network = StaticNetwork(contact_network,
                                    start_time,
                                    end_time)
        
        self.static_network_series[start_end_time] = new_network 

    def save_network_by_end_time(self,
                                 contact_network,
                                 end_time):
            
        start_end_time = self.StartEndTime(start=end_time-self.static_contact_interval, end=end_time)
        new_network = StaticNetwork(contact_network,
                                    start_time,
                                    end_time)
        
        self.static_network_series[start_end_time] = new_network 

    def save_end_statuses_to_network(self,
                                     end_time,
                                     end_statuses):
        start_end_time = next(filter(lambda keys: abs(keys.end - end_time) < 1e-8, self.static_network_series.keys()))
        self.static_network_series[start_end_time].set_end_statuses(end_statuses)
    
    def save_start_statuses_to_network(self,
                                       start_time,
                                       start_statuses):
        start_end_time =next(filter(lambda keys: abs(keys.start - start_time) < 1e-8, self.static_network_series.keys()))
        self.static_network_series[start_end_time].set_start_statuses(start_statuses)
                                  
    def get_network_from_start_time(self,
                                    start_time):
        start_end_time = next(filter(lambda keys: abs(keys.start - start_time) < 1e-8, self.static_network_series.keys()))
        return self.static_network_series[start_end_time]
  
    def get_network_from_end_time(self,
                                  end_time):
        start_end_time =next(filter(lambda keys: abs(keys.end - end_time) < 1e-8, self.static_network_series.keys()))
        return self.static_network_series[start_end_time]
    
