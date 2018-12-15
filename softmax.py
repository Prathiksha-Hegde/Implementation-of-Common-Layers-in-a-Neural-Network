# import the necessary libraries
import numpy as np


class SoftMaxLayer(object):
    def __init__(self):
  
        self.y = None

    def forward(self, x):
        """
        forward pass of softmax layer

        Parameters
       
        x : np.array- The input data of size number of training samples x number of features

        Returns- np.array - The output of the layer

        Stores - self.y : np.array - The output of the layer (needed for backpropagation)
        """
        
        max_x = np.max(x, axis=1)
        x_new = x - max_x.reshape((x.shape[0], 1))
        Nr = np.exp(x_new)
        Dr = np.sum(Nr, axis=1).reshape(x.shape[0], 1)
        self.y = Nr / Dr
        
        return self.y

    def backward(self, y_grad):
        """
        Backward computation of softmax

        Parameters - y_grad - np.array- The gradient at the output

        Returns - np.array - The gradient at the input

        """
        
        dlx = np.zeros([y_grad.shape[0], y_grad.shape[1]])
        
        for i in range(0, self.y.shape[0]):
            x = np.diag(self.y[i])
            dyx = x - np.outer(self.y[i].T, self.y[i])
            dlx[i] = np.dot(y_grad[i], dyx).reshape((1, self.y.shape[1]))
        return dlx
        



    def update_param(self, lr):
	# no learning for softmax layer
        pass  
