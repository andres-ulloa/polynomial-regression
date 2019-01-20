import math
import numpy as np

class model:
    
    def __init__ (self, weights, order, error_registry):

        self.weights = weights
        self.order = order
        self.error_registry = error_registry

    def sample(arg):

        dependent_variable = 0

        for i in range(0, len(weights)):
        
            if power_index > 1:
                dependent_variable += weights[i] * pow(arg, power_index)
            else:
                dependent_variable += weights[i]

            if power_index == order:

                feature_vector_index += 1
                power_index = 1    

        return dependent_variable