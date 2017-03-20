"""
This module implements the (location addressing only) Neural Turing Machine in TensorFlow.

NOTE ABOUT MATRICES: To follow the TF convention, we think of the matrix of a linear
map F: W -> V with dim(W) = m, dim(V) = n as a [m,n] tensor that is, a matrix with m rows
and n columns. The (i,j) entry of this matrix represents F(e_i)_j where e_i is the basis of W.
This explains, for example, the discrepancy between the rotation matrix R we write down here,
and the R in our paper. They are the same, once you learn to write matrices the "wrong" way. (Testing)
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
# Helper functions

def rotation_tensor(size,powers):
    """
    Returns rotation matrices as a [3,3,3] tensor, which is [R^{p_1}, R^{p_2}, ...]
    where R is the rotation matrix sending the first basis element to the second and 
    the final basis element to the first, and powers = [p_1,p_2,...]. The size of the
    matrices is given by "siez". Note the convention about matrices at the
    top of this file.
    """
    one_hots = []
    for i in range(size):
        a = [0.0]*size
        a[i] = 1.0
        one_hots.append(tf.constant(a))

    R_list = []
    for i in powers:
        R = []
        for j in range(size):
            index = (j + i) % size
            R.append(one_hots[index])
        R_list.append(tf.stack(R))

    R_tensor = tf.stack(R_list)

    return R_tensor

# The following are stolen from
# http://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model/38161314
def count_number_trainable_params():
    '''
    Counts the number of trainable variables.
    '''
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params

def get_nb_params_shape(shape):
    '''
    Computes the total number of params for a given shap.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    '''
    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params 
    
##################
# Standard RNN controller
# TODO(5-3-2017): not tested

class StandardRNN(RNNCell):
    """
    The main NTM code.
    """
    def __init__(self, num_units, input_size, activation=tanh):
        self._num_units = num_units
        self._activation = activation
        self._input_size = input_size
        
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
        with tf.variable_scope(scope,reuse=reuse): 
            # input has shape [batch_size, input_size]
            # state has shape [batch_size, state_size]

            H = tf.get_variable("H", [_num_units,_num_units])
            U = tf.get_variable("U", [_input_size,_num_units])
            B = tf.get_variable("B", [_num_units], initializer=init_ops.constant_initializer(0.0))
            
            state_new = self._activation(tf.matmul(state, H) + tf.matmul(input,U) + B)
            
        return state_new, state_new
        # the return is output, state
        
##################
# NTM Controller
    
class NTM(RNNCell):
    """
    The main NTM code.
    """
    def __init__(self, num_units, input_size, controller_state_size,
                memory_address_size,memory_content_size, powers):
        self._num_units = num_units
        self._input_size = input_size
        self._controller_state_size = controller_state_size
        self._memory_address_size = memory_address_size
        self._memory_content_size = memory_content_size
        self._powers = powers
        
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
            # Viewed as a matrix of shape [mas,mcs] the rows of M index memory locations.
            #
            # NOTE: these vectors are all batched, so in the following h0 has shape
            # [batch_size, controller_state_size], for example.
            
            css = self._controller_state_size
            mas = self._memory_address_size
            mcs = self._memory_content_size
            powers = self._powers # the powers of the rotation matrix we allow
            
            h0, r, w, M = tf.split(state, [css, mas, mas, mas * mcs], 1)
            
            init = init_ops.constant_initializer(0.0)
            perform_sharpening = True
            
            # Sharpening factor gamma, one for read and one for write
            W_gamma_read = tf.get_variable("W_gamma_read", [css,1])
            B_gamma_read = tf.get_variable("B_gamma_read", [], initializer=init)
            gamma_read = 1.0 + tf.nn.relu(tf.matmul(h0,W_gamma_read) + B_gamma_read) # shape [batch_size,1]
            
            W_gamma_write = tf.get_variable("W_gamma_write", [css,1])
            B_gamma_write = tf.get_variable("B_gamma_write", [], initializer=init)
            gamma_write = 1.0 + tf.nn.relu(tf.matmul(h0,W_gamma_write) + B_gamma_write) # shape [batch_size,1]
            
            # Now generate the s, q, e, a vectors
            W_s = tf.get_variable("W_s", [css,len(powers)])
            B_s = tf.get_variable("B_s", [len(powers)], initializer=init)
            s = tf.nn.softmax(tf.matmul(h0,W_s) + B_s) # shape [batch_size,len(powers)]

            W_q = tf.get_variable("W_q", [css,len(powers)])
            B_q = tf.get_variable("B_q", [len(powers)], initializer=init)
            q = tf.nn.softmax(tf.matmul(h0,W_q) + B_q) # shape [batch_size,len(powers)]

            W_e = tf.get_variable("W_e", [css,mcs])
            B_e = tf.get_variable("B_e", [mcs], initializer=init)
            e = tf.sigmoid(tf.matmul(h0,W_e) + B_e) # shape [batch_size,mcs]

            W_a = tf.get_variable("W_a", [css,mcs])
            B_a = tf.get_variable("B_a", [mcs], initializer=init)
            a = tf.nn.relu(tf.matmul(h0,W_a) + B_a) # shape [batch_size,mcs]
            
            # Add and forget on the memory
            M = tf.reshape(M, [-1, mas, mcs])
            erase_term = tf.matmul( M, tf.matrix_diag(e) ) # shape [batch_size, mas, mcs]
            add_term = tf.matmul( tf.reshape(w,[-1,mas,1]), tf.reshape(a,[-1,1,mcs]) ) # shape [batch_size, mas, mcs]
            M_new = M - erase_term + add_term
            M_new = tf.reshape(M_new, [-1, mas * mcs])
            
            # Do the rotations of the read and write addresses
            # r has shape [batch_size,mas]
            Rtensor = rotation_tensor(mas,powers)

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
            
            # Perform sharpening
            if( perform_sharpening == True ):
                sharpening_tensor_r = tf.zeros_like(r_new) + gamma_read
                sharp_r = tf.pow(r_new + 1e-6, sharpening_tensor_r)
                denom_r = tf.reduce_sum(sharp_r, axis=1, keep_dims=True)
                r_new = sharp_r / denom_r
            
                sharpening_tensor_w = tf.zeros_like(w_new) + gamma_write
                sharp_w = tf.pow(w_new + 1e-6, sharpening_tensor_w)
                denom_w = tf.reduce_sum(sharp_w, axis=1, keep_dims=True)
                w_new = sharp_w / denom_w

            # Construct new state
            H = tf.get_variable("H", [css,css])
            U = tf.get_variable("U", [self._input_size,css])
            B = tf.get_variable("B", [css], initializer=init)
        
            V = tf.get_variable("V", [mcs,css]) # converts from memory to controller state
            Mr = tf.matmul( M, tf.reshape(r,[-1,mas,1]), transpose_a=True )
            Mr = tf.reshape( Mr, [-1,mcs] )
            
            h0_new = tf.nn.tanh(tf.matmul(h0, H) + tf.matmul(Mr,V) + tf.matmul(input,U) + B)
        
            state_new = tf.concat([h0_new, r_new, w_new, M_new], 1)   
        return h0_new, state_new
        # the return is output, state
        
##################
# Pattern NTM Controller
    
class PatternNTM(RNNCell):
    """
    The main Pattern NTM code.
    """
    def __init__(self, num_units, input_size, controller_state_size,
                memory_address_size,memory_content_size, powers1, powers2):
        self._num_units = num_units
        self._input_size = input_size
        self._controller_state_size = controller_state_size
        self._memory_address_size = memory_address_size
        self._memory_content_size = memory_content_size
        self._powers1 = powers1
        self._powers2 = powers2
        
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
            # h = (h0,r1,w1,r2,w2,M1,M2)
            #
            # where
            #
            #   - h0 is the controller internal state (size controller_state_size)
            #   - ri is the ith read address weights (size memory_address_size)
            #   - wi is the ith write address weights (size memory_address_size)
            #   - Mi is the ith memory state (size memory_address_size*memory_content_size)
            #
            # Viewed as a matrix of shape [mas,mcs] the rows of M index memory locations.
            #
            # NOTE: these vectors are all batched, so in the following h0 has shape
            # [batch_size, controller_state_size], for example.
            
            css = self._controller_state_size
            mas = self._memory_address_size
            mcs = self._memory_content_size
            powers1 = self._powers1 # the powers of the rotation matrix we allow acting on ring 1
            powers2 = self._powers2 # the powers of the rotation matrix we allow acting on ring 2
            
            h0, r1, w1, r2, w2, M1, M2 = tf.split(state, [css, mas, mas, mas, mas, mas * mcs, mas * len(powers1)], 1)
            
            init = init_ops.constant_initializer(0.0)
            perform_sharpening = True
            
            # Note that M2 is [mas, mas]
            
            # Sharpening factor gamma, one for read and one for write, for each ring
            W_gamma_read1 = tf.get_variable("W_gamma_read1", [css,1])
            B_gamma_read1 = tf.get_variable("B_gamma_read1", [], initializer=init)
            gamma_read1 = 1.0 + tf.nn.relu(tf.matmul(h0,W_gamma_read1) + B_gamma_read1) # shape [batch_size,1]
            
            W_gamma_write1 = tf.get_variable("W_gamma_write1", [css,1])
            B_gamma_write1 = tf.get_variable("B_gamma_write1", [], initializer=init)
            gamma_write1 = 1.0 + tf.nn.relu(tf.matmul(h0,W_gamma_write1) + B_gamma_write1) # shape [batch_size,1]

            W_gamma_read2 = tf.get_variable("W_gamma_read2", [css,1])
            B_gamma_read2 = tf.get_variable("B_gamma_read2", [], initializer=init)
            gamma_read2 = 1.0 + tf.nn.relu(tf.matmul(h0,W_gamma_read2) + B_gamma_read2) # shape [batch_size,1]
            
            W_gamma_write2 = tf.get_variable("W_gamma_write2", [css,1])
            B_gamma_write2 = tf.get_variable("B_gamma_write2", [], initializer=init)
            gamma_write2 = 1.0 + tf.nn.relu(tf.matmul(h0,W_gamma_write2) + B_gamma_write2) # shape [batch_size,1]

            # Now generate the s, q, e, a vectors
            W_s1 = tf.get_variable("W_s1", [css,len(powers1)])
            B_s1 = tf.get_variable("B_s1", [len(powers1)], initializer=init)
            s1 = tf.nn.softmax(tf.matmul(h0,W_s1) + B_s1) # shape [batch_size,len(powers1)]

            W_s2 = tf.get_variable("W_s2", [css,len(powers2)])
            B_s2 = tf.get_variable("B_s2", [len(powers2)], initializer=init)
            s2 = tf.nn.softmax(tf.matmul(h0,W_s2) + B_s2) # shape [batch_size,len(powers2)]

            W_q2 = tf.get_variable("W_q2", [css,len(powers2)])
            B_q2 = tf.get_variable("B_q2", [len(powers2)], initializer=init)
            q2 = tf.nn.softmax(tf.matmul(h0,W_q2) + B_q2) # shape [batch_size,len(powers2)]

            W_e1 = tf.get_variable("W_e1", [css,mcs])
            B_e1 = tf.get_variable("B_e1", [mcs], initializer=init)
            e1 = tf.sigmoid(tf.matmul(h0,W_e1) + B_e1) # shape [batch_size,mcs]

            W_e2 = tf.get_variable("W_e2", [css,len(powers1)])
            B_e2 = tf.get_variable("B_e2", [len(powers1)], initializer=init)
            e2 = tf.sigmoid(tf.matmul(h0,W_e2) + B_e2) # shape [batch_size,len(powers1)]

            W_a1 = tf.get_variable("W_a1", [css,mcs])
            B_a1 = tf.get_variable("B_a1", [mcs], initializer=init)
            a1 = tf.nn.relu(tf.matmul(h0,W_a1) + B_a1) # shape [batch_size,mcs]

            W_a2 = tf.get_variable("W_a2", [css,len(powers1)])
            B_a2 = tf.get_variable("B_a2", [len(powers1)], initializer=init)
            a2 = tf.nn.relu(tf.matmul(h0,W_a2) + B_a2) # shape [batch_size,len(powers1)]

            # Add and forget on the memory
            M1 = tf.reshape(M1, [-1, mas, mcs])
            erase_term1 = tf.matmul( M1, tf.matrix_diag(e1) ) # shape [batch_size, mas, mcs]
            add_term1 = tf.matmul( tf.reshape(w1,[-1,mas,1]), tf.reshape(a1,[-1,1,mcs]) ) # shape [batch_size, mas, mcs]
            M1_new = M1 - erase_term1 + add_term1
            M1_new = tf.reshape(M1_new, [-1, mas * mcs])

            M2 = tf.reshape(M2, [-1, mas, len(powers1)])
            erase_term2 = tf.matmul( M2, tf.matrix_diag(e2) ) # shape [batch_size, mas, len(powers1)]
            add_term2 = tf.matmul( tf.reshape(w2,[-1,mas,1]), tf.reshape(a2,[-1,1,len(powers1)]) ) # shape [batch_size, mas, len(powers1)]
            M2_new = M2 - erase_term2 + add_term2
            M2_new = tf.reshape(M2_new, [-1, mas * len(powers1)])
            
            # Do the rotations of the read and write addresses
            # r has shape [batch_size,mas]
            Rtensor1 = rotation_tensor(mas,powers1)
            Rtensor2 = rotation_tensor(mas,powers2)
            
            # yields a tensor of shape [batch_size, mas, mas]
            # each row of which is \sum_i q_i R^i, and this batch
            # of matrices is then applied to r to generate r_new
            # NOTE: These are actually batch matmuls (tf.batch_matmul
            # went away with v1.0, matmul now does it automatically on the
            # first index)
            w1_new = tf.matmul( tf.reshape(w1, [-1,1,mas]),
                                tf.tensordot( s1, Rtensor1, [[1], [0]] ) )
                                
            r2_new = tf.matmul( tf.reshape(r2, [-1,1,mas]),
                                tf.tensordot( q2, Rtensor2, [[1], [0]] ) )
            w2_new = tf.matmul( tf.reshape(w2, [-1,1,mas]),
                                tf.tensordot( s2, Rtensor2, [[1], [0]] ) )

            # The new thing in the pattern NTM is 
            Mr2 = tf.matmul( M2, tf.reshape(r2,[-1,mas,1]), transpose_a=True )
            Mr2 = tf.reshape( Mr2, [-1,len(powers1)] )            
            r1_new = tf.matmul( tf.reshape(r1, [-1,1,mas]),
                                tf.tensordot( Mr2, Rtensor1, [[1], [0]] ) )
                                
            r1_new = tf.reshape( r1_new, [-1,mas] )
            w1_new = tf.reshape( w1_new, [-1,mas] )
            r2_new = tf.reshape( r2_new, [-1,mas] )
            w2_new = tf.reshape( w2_new, [-1,mas] )            

            # Perform sharpening
            if( perform_sharpening == True ):
                sharpening_tensor_r1 = tf.zeros_like(r1_new) + gamma_read1
                sharp_r1 = tf.pow(r1_new + 1e-6, sharpening_tensor_r1)
                denom_r1 = tf.reduce_sum(sharp_r1, axis=1, keep_dims=True)
                r1_new = sharp_r1 / denom_r1
            
                sharpening_tensor_w1 = tf.zeros_like(w1_new) + gamma_write1
                sharp_w1 = tf.pow(w1_new + 1e-6, sharpening_tensor_w1)
                denom_w1 = tf.reduce_sum(sharp_w1, axis=1, keep_dims=True)
                w1_new = sharp_w1 / denom_w1
                
                sharpening_tensor_r2 = tf.zeros_like(r2_new) + gamma_read2
                sharp_r2 = tf.pow(r2_new + 1e-6, sharpening_tensor_r2)
                denom_r2 = tf.reduce_sum(sharp_r2, axis=1, keep_dims=True)
                r2_new = sharp_r2 / denom_r2
            
                sharpening_tensor_w2 = tf.zeros_like(w2_new) + gamma_write2
                sharp_w2 = tf.pow(w2_new + 1e-6, sharpening_tensor_w2)
                denom_w2 = tf.reduce_sum(sharp_w2, axis=1, keep_dims=True)
                w2_new = sharp_w2 / denom_w2
                
            H = tf.get_variable("H", [css,css])
            U = tf.get_variable("U", [self._input_size,css])
            B = tf.get_variable("B", [css], initializer=init)
        
            V = tf.get_variable("V", [mcs,css]) # converts from memory to controller state
            Mr1 = tf.matmul( M1, tf.reshape(r1,[-1,mas,1]), transpose_a=True )
            Mr1 = tf.reshape( Mr1, [-1,mcs] )
            
            h0_new = tf.nn.tanh(tf.matmul(h0, H) + tf.matmul(Mr1,V) + tf.matmul(input,U) + B)
                    
            state_new = tf.concat([h0_new, r1_new, w1_new, r2_new, w2_new, M1_new, M2_new], 1)   
        return h0_new, state_new
        # the return is output, state
        
##################
# Pattern NTM Controller alterate
# This version allows the controller to also manipulate the read address
    
class PatternNTM_alt(RNNCell):
    """
    The main Pattern NTM code.
    """
    def __init__(self, num_units, input_size, controller_state_size,
                memory_address_size,memory_content_size, powers1, powers2, activation=tanh):
        self._num_units = num_units
        self._activation = activation
        self._input_size = input_size
        self._controller_state_size = controller_state_size
        self._memory_address_size = memory_address_size
        self._memory_content_size = memory_content_size
        self._powers1 = powers1
        self._powers2 = powers2
        
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
            # h = (h0,r1,w1,r2,w2,M1,M2)
            #
            # where
            #
            #   - h0 is the controller internal state (size controller_state_size)
            #   - ri is the ith read address weights (size memory_address_size)
            #   - wi is the ith write address weights (size memory_address_size)
            #   - Mi is the ith memory state (size memory_address_size*memory_content_size)
            #
            # Viewed as a matrix of shape [mas,mcs] the rows of M index memory locations.
            #
            # NOTE: these vectors are all batched, so in the following h0 has shape
            # [batch_size, controller_state_size], for example.
            
            css = self._controller_state_size
            mas = self._memory_address_size
            mcs = self._memory_content_size
            powers1 = self._powers1 # the powers of the rotation matrix we allow acting on ring 1
            powers2 = self._powers2 # the powers of the rotation matrix we allow acting on ring 2
            
            h0, r1, w1, r2, w2, M1, M2 = tf.split(state, [css, mas, mas, mas, mas, mas * mcs, mas * len(powers1)], 1)
            
            # Note that M2 is [mas, mas]
            
            # Now generate the s, q, e, a vectors
            W_s1 = tf.get_variable("W_s1", [css,len(powers1)])
            B_s1 = tf.get_variable("B_s1", [len(powers1)], initializer=init_ops.constant_initializer(0.0))
            s1 = tf.nn.softmax(tf.matmul(h0,W_s1) + B_s1) # shape [batch_size,len(powers1)]

            W_s2 = tf.get_variable("W_s2", [css,len(powers2)])
            B_s2 = tf.get_variable("B_s2", [len(powers2)], initializer=init_ops.constant_initializer(0.0))
            s2 = tf.nn.softmax(tf.matmul(h0,W_s2) + B_s2) # shape [batch_size,len(powers2)]

            W_q1 = tf.get_variable("W_q1", [css,len(powers1)])
            B_q1 = tf.get_variable("B_q1", [len(powers1)], initializer=init_ops.constant_initializer(0.0))
            q1 = tf.nn.softmax(tf.matmul(h0,W_q1) + B_q1) # shape [batch_size,len(powers1)]

            W_q2 = tf.get_variable("W_q2", [css,len(powers2)])
            B_q2 = tf.get_variable("B_q2", [len(powers2)], initializer=init_ops.constant_initializer(0.0))
            q2 = tf.nn.softmax(tf.matmul(h0,W_q2) + B_q2) # shape [batch_size,len(powers2)]

            W_e1 = tf.get_variable("W_e1", [css,mcs])
            B_e1 = tf.get_variable("B_e1", [mcs], initializer=init_ops.constant_initializer(0.0))
            e1 = tf.nn.relu(tf.matmul(h0,W_e1) + B_e1) # shape [batch_size,mcs]

            W_e2 = tf.get_variable("W_e2", [css,len(powers1)])
            B_e2 = tf.get_variable("B_e2", [len(powers1)], initializer=init_ops.constant_initializer(0.0))
            e2 = tf.nn.relu(tf.matmul(h0,W_e2) + B_e2) # shape [batch_size,len(powers1)]

            W_a1 = tf.get_variable("W_a1", [css,mcs])
            B_a1 = tf.get_variable("B_a1", [mcs], initializer=init_ops.constant_initializer(0.0))
            a1 = tf.nn.relu(tf.matmul(h0,W_a1) + B_a1) # shape [batch_size,mcs]

            W_a2 = tf.get_variable("W_a2", [css,len(powers1)])
            B_a2 = tf.get_variable("B_a2", [len(powers1)], initializer=init_ops.constant_initializer(0.0))
            a2 = tf.nn.relu(tf.matmul(h0,W_a2) + B_a2) # shape [batch_size,len(powers1)]

            # Add and forget on the memory
            M1 = tf.reshape(M1, [-1, mas, mcs])
            erase_term1 = tf.matmul( M1, tf.matrix_diag(e1) ) # shape [batch_size, mas, mcs]
            add_term1 = tf.matmul( tf.reshape(w1,[-1,mas,1]), tf.reshape(a1,[-1,1,mcs]) ) # shape [batch_size, mas, mcs]
            M1_new = M1 - erase_term1 + add_term1
            M1_new = tf.reshape(M1_new, [-1, mas * mcs])

            M2 = tf.reshape(M2, [-1, mas, len(powers1)])
            erase_term2 = tf.matmul( M2, tf.matrix_diag(e2) ) # shape [batch_size, mas, len(powers1)]
            add_term2 = tf.matmul( tf.reshape(w2,[-1,mas,1]), tf.reshape(a2,[-1,1,len(powers1)]) ) # shape [batch_size, mas, len(powers1)]
            M2_new = M2 - erase_term2 + add_term2
            M2_new = tf.reshape(M2_new, [-1, mas * len(powers1)])
            
            # Do the rotations of the read and write addresses
            # r has shape [batch_size,mas]
            Rtensor1 = rotation_tensor(mas,powers1)
            Rtensor2 = rotation_tensor(mas,powers2)
            
            # yields a tensor of shape [batch_size, mas, mas]
            # each row of which is \sum_i q_i R^i, and this batch
            # of matrices is then applied to r to generate r_new
            # NOTE: These are actually batch matmuls (tf.batch_matmul
            # went away with v1.0, matmul now does it automatically on the
            # first index)
            r1_new = tf.matmul( tf.reshape(r1, [-1,1,mas]),
                                tf.tensordot( q1, Rtensor1, [[1], [0]] ) )
            w1_new = tf.matmul( tf.reshape(w1, [-1,1,mas]),
                                tf.tensordot( s1, Rtensor1, [[1], [0]] ) )
                                
            r2_new = tf.matmul( tf.reshape(r2, [-1,1,mas]),
                                tf.tensordot( q2, Rtensor2, [[1], [0]] ) )
            w2_new = tf.matmul( tf.reshape(w2, [-1,1,mas]),
                                tf.tensordot( s2, Rtensor2, [[1], [0]] ) )

            # The new thing in the pattern NTM is 
            Mr2 = tf.matmul( M2, tf.reshape(r2,[-1,mas,1]), transpose_a=True )
            Mr2 = tf.reshape( Mr2, [-1,len(powers1)] )            
            r1_new = tf.matmul( tf.reshape(r1_new, [-1,1,mas]),
                                tf.tensordot( Mr2, Rtensor1, [[1], [0]] ) )
                                
            r1_new = tf.reshape( r1_new, [-1,mas] )
            w1_new = tf.reshape( w1_new, [-1,mas] )
            r2_new = tf.reshape( r2_new, [-1,mas] )
            w2_new = tf.reshape( w2_new, [-1,mas] )            

            H = tf.get_variable("H", [css,css])
            U = tf.get_variable("U", [self._input_size,css])
            B = tf.get_variable("B", [css], initializer=init_ops.constant_initializer(0.0))
        
            V = tf.get_variable("V", [mcs,css]) # converts from memory to controller state
            Mr1 = tf.matmul( M1, tf.reshape(r1,[-1,mas,1]), transpose_a=True )
            Mr1 = tf.reshape( Mr1, [-1,mcs] )
            
            h0_new = self._activation(tf.matmul(h0, H) + tf.matmul(Mr1,V) + tf.matmul(input,U) + B)
                    
            state_new = tf.concat([h0_new, r1_new, w1_new, r2_new, w2_new, M1_new, M2_new], 1)   
        return h0_new, state_new
        # the return is output, state
