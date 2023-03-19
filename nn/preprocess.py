# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    pass

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """

    #store list of encoded sequences 
    encoded_allseq=[]
    
    nuc_dict={'A':[1,0,0,0], 
              'T':[0,1,0,0], 
              'C':[0,0,0,1], 
              'G':[0,0,0,1]}
    
    

    for i, seq in enumerate(seq_arr):
        
        #get list of encodings from dict
        encode_per_nuc=([nuc_dict[n] for n in seq.upper() if n in nuc_dict])
        #flatten and add to master list 
        encode_seq= [i for per_nuc in encode_per_nuc for i in per_nuc]
        encoded_allseq.append(encode_seq)

    return(encoded_allseq)