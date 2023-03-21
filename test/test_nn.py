# TODO: import dependencies and write unit tests below
import pytest 
import numpy as np
from sklearn.metrics import log_loss, mean_squared_error


from nn.nn import NeuralNetwork
from nn.preprocess import sample_seqs, one_hot_encode_seqs
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



#make an instance of nn class to be used by multiple functions
example_nn = NeuralNetwork(nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
                                      {'input_dim': 16, 'output_dim': 64, 'activation': 'relu'}],
                            lr = 0.0005, seed = 3, batch_size = 100, epochs = 100, loss_function='mse', verbose=False)


#use digits as a test dataset
digits = load_digits()
  
#split into train and test sets 
X_train, X_test, y_train, y_test=train_test_split(digits.data, digits.target, train_size=0.8, random_state=3)



def test_single_forward():

    #check that A_curr and Z_cur values are expected as to when calculated by hand
    W_curr=np.array([[1,1,1], [1,2,3]])
    b_curr=np.array([[0,0]])
    A_prev=np.array([1,2,3])
    A_curr, Z_curr=example_nn._single_forward(W_curr, b_curr, A_prev, activation='relu')
    print(A_curr, Z_curr)


    assert np.isclose(A_curr, np.array([]))
    #assert np.isclose(Z_curr, [])
    

    #check dimensions of A_curr and Z_curr

 
    

def test_forward():


    #check that 
    pass

def test_single_backprop():
    pass

def test_predict():

    #test that prediction is correct 


    #use TF data to access accuracy 


    # #make logreg class 
	# log_model = LogisticRegressor(num_feats = len(w) - 1, learning_rate=0.01, max_iter=5000, batch_size=50)

	# #train logreg model 
	# log_model.train_model(X_train, y_train, X_val, y_val)
	
	# #make predictions on new test set 
	# model_pred = log_model.make_prediction(X_test)
	# model_pred_binary = np.where(model_pred > 0.5, 1, 0)
	# y_test_binary=np.where(y_test > 0.5, 1, 0)

	# #get accuracy of prediction on separate test set 
	# accuracy=np.sum(model_pred_binary == y_test_binary)/len(y_test_binary)

	# #check accuracy is at least a little better than chance lol
	# assert accuracy>0.6, "accuracy is worse than 0.6!"
        


    pass

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
    y_pred=[0.2, 0.9, 1]
        

    #test with nn model
    nn_backprop=example_nn._binary_cross_entropy_backprop(np.array(y_true), np.array(y_pred))
    print(nn_backprop)
    assert np.isclose(nn_backprop, -1.120375, rtol=1e-4), 'BCE backpropagation value is not as expected'


def test_mean_squared_error():

    #make random y_true and y_pred
    y_true=[0, 1, 0.75, 0.25]
    y_pred=[0.1, 0.9, 0.7, 0.2]


    #compare to hand calculated 
    #nn_mse=example_nn._mean_squared_error(np.expand_dims(y_true, 1), np.expand_dims(y_pred,1))
    nn_mse=example_nn._mean_squared_error(np.array(y_true), np.array(y_pred))
    sklearn_mse=mean_squared_error(y_true, y_pred)

	#check that mse  from our function is close to sklearn mse with reasonable tolerance 
    assert np.isclose(nn_mse, sklearn_mse, rtol=1e-4), 'MSE is not as expected'



def test_mean_squared_error_backprop():
    #make random y_true and y_pred
    y_true=[0, 1, 4]
    y_pred=[1, 2, 3]


    #compare to hand calculated 
    #nn_mse=example_nn._mean_squared_error(np.expand_dims(y_true, 1), np.expand_dims(y_pred,1))
    nn_mse=example_nn._mean_squared_error_backprop(np.array(y_true), np.array(y_pred))
    print(nn_mse)

	#check that mse  from our function is close to sklearn mse with reasonable tolerance 
    assert np.isclose(nn_mse, 2/3, rtol=1e-4), 'MSE backpropagation value is not as expected'


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
