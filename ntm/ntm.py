"""
This module implements the (location addressing only) Neural Turing Machine in TensorFlow.

NOTE ABOUT MATRICES: To follow the TF convention, we think of the matrix of a linear
map F: W -> V with dim(W) = m, dim(V) = n as a [m,n] tensor that is, a matrix with m rows
and n columns. The (i,j) entry of this matrix represents F(e_i)_j where e_i is the basis of W.
This explains, for example, the discrepancy between the rotation matrix R we write down here,
and the R in our paper. They are the same, once you learn to write matrices the "wrong" way.
"""

# The next three lines are recommend by TF
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import collections
import six
import math
import time

from random import shuffle
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

##################
# NTM Controller

def rotation_tensor(size):
    """
    Returns rotation matrices as a [3,3,3] tensor, which is [R^0, R^1, ..., R^{size-1}] 
    where R is the rotation matrix sending the first basis element to the second and 
    the final basis element to the first.
    """
    one_hots = []
    for i in range(size):
        a = [0.0]*size
        a[i] = 1.0
        one_hots.append(tf.constant(a))

    R_list = []
    for i in range(size):
        R = []
        for j in range(size):
            index = (j + i) % size
            R.append(one_hots[index])
        R_list.append(tf.stack(R))

    R_tensor = tf.stack(R_list)

    return R_tensor
    
class NTMRNNCell(RNNCell):
    """
    The main NTM code.
    """
    def __init__(self, num_units, input_size, controller_state_size,
                memory_address_size,memory_content_size, activation=tanh):
        self._num_units = num_units
        self._activation = activation
        self._input_size = input_size
        self._controller_state_size = controller_state_size
        self._memory_address_size = memory_address_size
        self._memory_content_size = memory_content_size
        
    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units
        
    @property
    def input_size(self):
        return self._input_size

    def __call__(self, input, state, scope, reuse=True):
        # the scope business gives a namespace to our weight variable matrix names
        with tf.variable_scope(scope,reuse=reuse): 
            # input has shape [batch_size, input_size]
            # state has shape [batch_size, state_size]
            
            # We divide up a state vector as follows:
            #
            # h = (h0,r,w,M)
            #
            # where
            #
            #   - h0 is the controller internal state (size controller_state_size)
            #   - r is the read address weights (size memory_address_size)
            #   - w is the write address weights (size memory_address_size)
            #   - M is the memory state (size memory_address_size*memory_content_size)
            #
            # the memory state vector M is the concatenation of the vectors at each
            # memory location, in order. Viewed as a matrix of shape [mas,mcs] the
            # rows index memory locations.
            #
            # NOTE: these vectors are all batched, so in the following h0 has shape
            # [batch_size, controller_state_size], for example.
            
            css = self._controller_state_size
            mas = self._memory_address_size
            mcs = self._memory_content_size
            
            # The first step is to decompose the state vector
            h0, r, w, M = tf.split(state, [css, mas, mas, mas * mcs], 1)
            
            # Now generate the s, q, e, a vectors
            W_s = tf.get_variable("W_s", [css,mas])
            B_s = tf.get_variable("B_s", [mas])
            s = tf.nn.softmax(tf.matmul(h0,W_s) + B_s) # shape [batch_size,mas]

            W_q = tf.get_variable("W_q", [css,mas])
            B_q = tf.get_variable("B_q", [mas])
            q = tf.nn.softmax(tf.matmul(h0,W_q) + B_q) # shape [batch_size,mas]

            W_e = tf.get_variable("W_e", [css,mcs])
            B_e = tf.get_variable("B_e", [mcs])
            e = tf.nn.softmax(tf.matmul(h0,W_e) + B_e) # shape [batch_size,mcs]

            W_a = tf.get_variable("W_a", [css,mcs])
            B_a = tf.get_variable("B_a", [mcs])
            a = tf.nn.softmax(tf.matmul(h0,W_a) + B_a) # shape [batch_size,mcs]

            # Add and forget on the memory
            # TODO: not sure if matrix_diag is slow
            M = tf.reshape(M, [-1, mas, mcs])
            erase_term = tf.matmul( M, tf.matrix_diag(e) ) # shape [batch_size, mas, mcs]
            add_term = tf.matmul( tf.reshape(w,[-1,mas,1]), tf.reshape(a,[-1,1,mcs]) ) # shape [batch_size, mas, mcs]
            M_new = M - erase_term + add_term
            M_new = tf.reshape(M_new, [-1, mas * mcs])
            
            # Do the rotations of the read and write addresses
            # r has shape [batch_size,mas]
            Rtensor = rotation_tensor(mas)

            # yields a tensor of shape [batch_size, mas, mas]
            # each row of which is \sum_i q_i R^i, and this batch
            # of matrices is then applied to r to generate r_new
            # NOTE: These are actually batch matmuls (tf.batch_matmul
            # went away with v1.0, matmul now does it automatically on the
            # first index)
            r_new = tf.matmul( tf.reshape(r, [-1,1,mas]),
                                tf.tensordot( q, Rtensor, [[1], [0]] ) )
            w_new = tf.matmul( tf.reshape(w, [-1,1,mas]),
                                tf.tensordot( s, Rtensor, [[1], [0]] ) )
                                
            r_new = tf.reshape( r_new, [-1,mas] )
            w_new = tf.reshape( w_new, [-1,mas] )

            H = tf.get_variable("H", [css,css])
            U = tf.get_variable("U", [self._input_size,css])
            B = tf.get_variable("B", [css], initializer=init_ops.constant_initializer(0.0))
        
            V = tf.get_variable("V", [mcs,css]) # converts from memory to controller state
            Mr = tf.matmul( M, tf.reshape(r,[-1,mas,1]), transpose_a=True )
            Mr = tf.reshape( Mr, [-1,mcs] )
            
            h0_new = self._activation(tf.matmul(h0, H) + tf.matmul(Mr,V) + tf.matmul(input,U) + B)
        
            #h0_new = self._activation(tf.matmul(h0, H) + tf.matmul(input,U) + B)
            
            state_new = tf.concat([h0_new, r_new, w_new, M_new], 1)   
        return h0_new, state_new
        # the return is output, state