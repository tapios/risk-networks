import numpy as np

class Transform:

    def __init__(
            self,
            name="identity",
            **kwargs):   
        '''
        Instantiate the object to implement transforms defined by a name
        
        Args
        ----
        name (string), extra parameters,
        "identity",    None
        "logit",       None
        "tanh",        lengthscale
                       
        
        '''
        self.name = name
        if self.name == "tanh":
            self.lengthscale = kwargs.get('lengthscale',1) #1 is a default lengthscale
    
    def apply_transform(self,x):
        return {   
            'identity' : lambda x: x,
            'logit'    : lambda x: np.log(np.maximum(x, 1e-9) / np.maximum(1.0 - x, 1e-9)),
            'tanh'     : lambda x: np.arctanh(2*(x - 1))*self.lengthscale
        }[self.name](x)

    def apply_inverse_transform(self,x):
        return {
            'identity' : lambda x: x,
            'logit'    : lambda x: np.exp(x)/(np.exp(x) + 1.0),
            'tanh'     : lambda x: 1 + 0.5*np.tanh(x/self.lengthscale)
        }[self.name](x)

        
