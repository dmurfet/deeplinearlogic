"""
Helper routines involving sequences.

All sequences are in the alphabet of one-hot vectors of some size.
"""

from random import shuffle
import numpy as np

def one_hot_vectors(num_classes):
    one_hots = []
    for i in range(num_classes):
        a = [0.0]*num_classes
        a[i] = 1.0
        one_hots.append(np.array(a))
    return one_hots

def shuffled_binary_seqs(N):
    """
    Creates a shuffled list of all binary sequences of length N.
    """
    s = '{0:0' + str(N) + 'b}'
    seq_input = [s.format(i) for i in range(2**N)]
    shuffle(seq_input)
    seq_input = [map(int,i) for i in seq_input]
    ti = []
    for i in seq_input:
        temp_list = []
        for j in i:
            temp_list.append(j)
        ti.append(temp_list)
    return seq_input
    
def shuffled_seqs(N,num_classes):
    """
    Creates a shuffled list of all sequences of length N from the set 0,..,num_classes-1
    """
    seq_unshuffled = []
    a = [0]*N
    inc_index = N - 1
    
    while(1):
        seq_unshuffled.append(a)
        a = list(a) # make a copy
        j = N - 1
        
        while( j >= 0 and a[j] == num_classes - 1 ):
            a[j] = 0
            j = j - 1
            
        if( j == -1 ):
            break
            
        a[j] = a[j] + 1
    
    seq_shuffled = shuffle(seq_unshuffled)
    return seq_unshuffled