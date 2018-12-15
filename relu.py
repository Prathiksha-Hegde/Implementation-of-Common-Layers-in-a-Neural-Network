
# import libraries
import numpy as np


class ReluLayer(object):
    def __init__(self): 
        self.y = None


    def forward(self, x):
	
    """	
    Forward pass of Relu
	The foward function is:
	y = x if x > 0
	y = 0 otherwise

	Input - x - np.array: The input data of size number of training samples x number of features
  	Output - self.y - np.array - The output data stored for back propagation 
    """
        y=x>0
        self.y=y*x
        return self.y

    def backward(self, y_grad):

        """
        Implement backward pass of Relu
	
        Input - y_grad : np.array - The gradient at the output
        Ouput - np.array - The gradient at the input   
	
        """
        y=self.y>0
        grad=y_grad*y
        return grad


    def update_param(self, lr):
	# There are no parameters to update
        pass  
