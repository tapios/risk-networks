import numpy as np

class Transform:

    def __init__(
            self,
            name="identity"):   
        '''
        Instantiate the object to implement transforms defined by a name
        
        Args
        ----
        name (string): "identity", "logit"
        
        '''
        self.name = name
    
    def apply_transform(self,x):
        return {   
            'identity' : lambda x: x,
            'logit'    : lambda x:  np.log(np.maximum(x, 1e-9) / np.maximum(1.0 - x, 1e-9))
        }[self.name](x)

    def apply_inverse_transform(self,x):
        return {
            'identity' : lambda x: x,
            'logit'    : lambda x: np.exp(x_logit)/(np.exp(x_logit) + 1.0)
        }[self.name](x)

        
