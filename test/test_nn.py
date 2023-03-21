# TODO: import dependencies and write unit tests below
import pytest 
import numpy as np
from sklearn.metrics import log_loss, mean_squared_error

from nn.nn import NeuralNetwork
from nn.preprocess import sample_seqs, one_hot_encode_seqs



#make a toy instance of nn class to be used for testing 
example_nn = NeuralNetwork(nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
                                      {'input_dim': 16, 'output_dim': 64, 'activation': 'relu'}],
                            lr = 0.0005, seed = 3, batch_size = 11, epochs = 1, loss_function='mse', verbose=False)



def test_single_forward():

    #make some toy values
    W_curr=np.array([[1,1,1], [1,2,-2]])
    b_curr=np.array([[1,0,0]])
    A_prev=np.array([1,-2,1])
    
    #run single forward 
    A_curr, Z_curr=example_nn._single_forward(W_curr, b_curr, A_prev, activation='relu')
    print(A_curr)


   #check values to hand calculated 
    assert np.allclose(Z_curr, np.array([[1,-4], [0,-5], [0,-5]]))
    assert np.allclose(A_curr, np.array([[1,0], [0,0], [0,0]]))
 
    #check shape of matrices 
    assert A_curr.shape==(3,2), 'shape of A_curr is incorrect'
    assert Z_curr.shape==(3,2), 'shape of Z_curr is incorrect'

    #test nonexistent activation 
    with pytest.raises(Exception):
       example_nn._single_forward(W_curr, b_curr, A_prev, activation='idk'), 'forward is not raising error for nonexistent activation function!'
 

 
    

def test_forward():

    #make simple nn
    nn_arch_simple= [{'input_dim': 3, 'output_dim': 2, 'activation': 'relu'},
                                      {'input_dim': 2, 'output_dim': 3, 'activation': 'relu'}]
    simple_nn = NeuralNetwork(nn_arch = nn_arch_simple, lr = 0.0005, seed = 3, 
                               batch_size = 10, epochs = 1, loss_function='mse', verbose=False)


    #init some params 
    param_dict = {}
    #set seed so values are the same 
    np.random.seed(4)
    for idx, layer in enumerate(nn_arch_simple):
        layer_idx = idx + 1
        input_dim = layer['input_dim']
        output_dim = layer['output_dim']
        param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
        param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
    simple_nn._param_dict=param_dict


    #make toy X and run forward
    X=np.array([[1,2,5], [1,1,1]])
    y_pred, cache = simple_nn.forward(X)
   
    #check that one forward pass values are as expected 
    assert np.allclose(y_pred,np.array([[0, 0.035099, 0], [0, 0.035099, 0]]), rtol=1e-4)
    assert y_pred.shape==X.shape

    #check that cache values and shapes were set appropriately
    assert np.allclose(cache['A0'], X)
    assert np.allclose(cache['A1'],np.array([[0, 0], [0, 0]]))
    assert cache['Z1'].shape==(2,2)





def test_single_backprop():

    #make some toy values
    W_curr=np.array([[1,1], [1,2]])
    b_curr=np.array([[1]])
    A_prev=np.array([1,-2])

    Z_curr=np.array([2,-2])
    A_curr=np.array([2,-2])
    dA_curr=np.array([1,-1])
    
    #run single backprop
    dA_prev, dW_curr, db_curr=example_nn._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, 'relu')

  #check values to hand calculated 
    assert np.allclose(dA_prev, np.array([[1,1]]))
    assert np.allclose(dW_curr, np.array([[1]]))
    assert np.allclose(db_curr, np.array([[1]]))

    #test nonexistent activation 
    with pytest.raises(Exception):
       example_nn._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, 'idk'), 'backprop is not raising error for nonexistent activation function!'
 



def test_predict():

    #similar to test_forward above 

    #make simple nn 
    nn_arch_simple= [{'input_dim': 3, 'output_dim': 2, 'activation': 'relu'},
                                      {'input_dim': 2, 'output_dim': 3, 'activation': 'relu'}]
    simple_nn = NeuralNetwork(nn_arch = nn_arch_simple, lr = 0.0005, seed = 3, 
                               batch_size = 10, epochs = 1, loss_function='mse', verbose=False)


    #init some params 
    param_dict = {}
    #set seed so values are the same 
    np.random.seed(4)
    for idx, layer in enumerate(nn_arch_simple):
        layer_idx = idx + 1
        input_dim = layer['input_dim']
        output_dim = layer['output_dim']
        param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
        param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
    simple_nn._param_dict=param_dict
    #set initial param dict, which throws error if predict is run and params don't change
    simple_nn._param_dict_init=param_dict


    #make toy X 
    X=np.array([[1,2,5], [1,1,1]])

    #check that warning is thrown if fit isn't run before predict
    with pytest.raises(Warning):
        simple_nn.predict(X)
    
    #now just set init params to empty dict
    simple_nn._param_dict_init={}

    #check that predicted values are as expected 
    y_pred = simple_nn.predict(X)

    assert np.allclose(y_pred,np.array([[0, 0.035099, 0], [0, 0.035099, 0]]), rtol=1e-4)
    assert y_pred.shape==X.shape

    

 

def test_binary_cross_entropy():
        
    
    #make random y_true and y_pred
    y_true=[0, 1, 1, 0, 1, 1, 0]
    y_pred=[0.1, 0.9, 0.7, 0.2, 0.5, 0.6, 0.3]
        

    #test with nn model
    nn_loss=example_nn._binary_cross_entropy(np.array(y_true), np.array(y_pred))
    sklearn_loss=log_loss(y_true, y_pred)

	#check that bce loss from our function is close to sklearn loss with reasonable tolerance 
    assert np.isclose(nn_loss, sklearn_loss, rtol=1e-4), 'BCE is not as expected'
 


def test_binary_cross_entropy_backprop():

    #do vs hand calculated 

    y_true=[0, 1, 1]
    y_pred=[0.2, 0.5, 1]
        

    #test with nn model
    nn_backprop=example_nn._binary_cross_entropy_backprop(np.array(y_true), np.array(y_pred))
    assert np.allclose(nn_backprop, [0.41665, -2/3, -1/3], rtol=1e-4), 'BCE backpropagation value is not as expected'



def test_mean_squared_error():

    #make random y_true and y_pred
    y_true=[0, 1, 0.75, 0.25]
    y_pred=[0.1, 0.9, 0.7, 0.2]


    #compare to hand calculated 
    nn_mse=example_nn._mean_squared_error(np.array(y_true), np.array(y_pred))
    sklearn_mse=mean_squared_error(y_true, y_pred)

	#check that mse  from our function is close to sklearn mse with reasonable tolerance 
    assert np.isclose(nn_mse, sklearn_mse, rtol=1e-4), 'MSE is not as expected'



def test_mean_squared_error_backprop():
    #make random y_true and y_pred
    y_true=[0, 1, 4]
    y_pred=[1, 2, 3]


    #compare to hand calculated 
    nn_mse=example_nn._mean_squared_error_backprop(np.array(y_true), np.array(y_pred))
    print(nn_mse)

	#check that mse from our function is close to sklearn mse with reasonable tolerance 
    assert np.allclose(nn_mse, [2/3, 2/3, -2/3], rtol=1e-4), 'MSE backpropagation value is not as expected'


def test_sample_seqs():
    
    #make toy sequences and labels
    #assumes that sequences are all the same size 
    toy_seqs=['ATCG', 'ACTG', 'TTTT', 'ATAT', 'GTGT', 'ACTG', 'CTGA']
    toy_labels=['A', 'A', 'A', 'A', 'A', 'B', 'B']

    #resample
    sampled_seqs, sampled_labels=sample_seqs(toy_seqs, toy_labels)

    assert len(sampled_seqs)==10, 'length of total sampled seqs is incorrect'
    assert sampled_labels.count('A') ==sampled_labels.count('B'), 'different number of sampled sequences for labels '



def test_one_hot_encode_seqs():

   
    #test list of sequences 
    sequence_list=['ATCG', 'ACTA', 'TGTC']
    encoded_list=one_hot_encode_seqs(sequence_list)
    expected_list=np.array([[1,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0],
                            [1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0],
                             [0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,]])


    #assert one hot encodings are working 
    assert np.allclose(encoded_list,expected_list), 'one hot encodings are wrong!'
    assert encoded_list.shape[0]==len(sequence_list), 'dimension 0 of one hot encodings is wrong size! '
    assert encoded_list.shape[1]==len(sequence_list[1])*4, 'dimension 1 of one hot encodings is wrong size! '


    #test sequences with extraneous nucleotide
    with pytest.raises(AssertionError):
        wrong_nucleotide=['ATBCRTGT']
        one_hot_encode_seqs(wrong_nucleotide), 'model is not raising error for non ACGT nucleotides!'
