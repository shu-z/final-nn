# TODO: import dependencies and write unit tests below
import pytest 
import numpy as np
from sklearn.metrics import log_loss


from nn.nn import NeuralNetwork
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt



#make an instance of nn class to be used by multiple functions
example_nn = NeuralNetwork(nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
                                      {'input_dim': 16, 'output_dim': 64, 'activation': 'relu'}],
                            lr = 0.0005, seed = 3, batch_size = 100, epochs = 500, loss_function='mse')


#make lame test dataset to use 


def test_single_forward():
    pass

def test_forward():
    pass

def test_single_backprop():
    pass

def test_predict():

    #test that prediction is correct 
    pass

def test_binary_cross_entropy():
        
    
    #make random y_true and y_pred
    y_true=[0, 1, 1, 0, 1, 1, 0]
    y_pred=[0.1, 0.9, 0.7, 0.2, 0.5, 0.6, 0.3]
        

    #test with nn model
    nn_loss=example_nn._binary_cross_entropy(np.array(y_true), np.array(y_pred))
    sklearn_loss=log_loss(y_true, y_pred)

	#check that bce loss from our function is close to sklearn loss with reasonable tolerance 
    assert np.isclose(nn_loss, sklearn_loss, rtol=1e-4), 'BCE loss is not as expected'
 

def test_binary_cross_entropy_backprop():

    #do vs hand calculated 
    pass

def test_mean_squared_error():

    #make random y_true and y_pred
    y_true=[0, 1, 1, 0, 1, 1, 0]
    y_pred=[0.1, 0.9, 0.7, 0.2, 0.5, 0.6, 0.3]


    pass

def test_mean_squared_error_backprop():
    pass

def test_sample_seqs():
    pass

def test_one_hot_encode_seqs():
    pass