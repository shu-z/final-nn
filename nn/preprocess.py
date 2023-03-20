# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import random

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
    label_names, label_counts = np.unique(labels, return_counts=True)
    #print(label_counts)
    #print(label_names)
    
    label0 = [seq for seq, label in zip(seqs, labels) if label==label_names[0]]
    label1 = [seq for seq, label in zip(seqs, labels) if label==label_names[1]]
        
  
    
    if label_counts[0]==label_counts[1]:
        return(seqs, labels)
    
    
    elif label_counts[0]>label_counts[1]:
        
        #sample more of label_names[1] 
        sampled_label1=random.choices(label1, k=label_counts[0])
        #sampled_label1=random.choices(range(0, len(label1)), k=label_counts[0])

        sampled_seqs=label0 + sampled_label1
        sampled_labels=([label_names[0]] * label_counts[0]) + ([label_names[1]]* label_counts[0])
        
        return(sampled_seqs, sampled_labels)
        
        
    
    elif label_counts[0]<label_counts[1]:
        #sample more of label_names[0]
        
        sampled_label0=random.choices(label0, k=label_counts[1])
        #sampled_label1=random.choices(range(0, len(label1)), k=label_counts[0])

        sampled_seqs=sampled_label0 + label1
        sampled_labels=([label_names[0]] * label_counts[1]) + ([label_names[1]] * label_counts[1])

        return(sampled_seqs, sampled_labels)
        
        
    
    
    
    

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
    
        encode_per_nuc=([nuc_dict[n] for n in seq.upper() if n in nuc_dict])
        encoded_allseq.append(np.concatenate(encode_per_nuc))

    return(np.stack(encoded_allseq))