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