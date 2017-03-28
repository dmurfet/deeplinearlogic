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
    Returns rotation matrices as a [?,size,size] tensor, which is [R^{p_1}, R^{p_2}, ...]
    where R is the rotation matrix sending the first basis element to the second and 
    the final basis element to the first, and powers = [p_1,p_2,...]. The size of the
    matrices is given by "size". Note the convention about matrices at the
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
#
# The parameter use_direct_access controls whether the controller is able to
# manipulate the read address of the first memory ring directly.
    
class PatternNTM(RNNCell):
    """
    The main Pattern NTM code.
    """
    def __init__(self, num_units, input_size, controller_state_size,
                memory_address_sizes,memory_content_sizes, 
                powers, powers21):
        self._num_units = num_units
        self._input_size = input_size
        self._controller_state_size = controller_state_size
        self._memory_address_sizes = memory_address_sizes
        self._memory_content_sizes = memory_content_sizes
        self._powers = powers
        self._powers21 = powers21
        
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

            css = self._controller_state_size
            mas = self._memory_address_sizes
            mcs = self._memory_content_sizes
            num_rings = len(mas)
                        
            # We divide up a state vector as follows:
            #
            # h = (h0,r1,w1,...,rN,wN,M1,M2,...,MN)
            #
            # where
            #
            #   - h0 is the controller internal state (size css)
            #   - ri is the ith read address weights (size mas[i])
            #   - wi is the ith write address weights (size mcs[i])
            #   - Mi is the ith memory state (size mas[i]*mcs[i])
            #
            # Viewed as a matrix of shape [mas[i],mcs[i]] the rows of M[i] index memory locations.
            #
            # NOTE: these vectors are all batched, so in the following h0 has shape
            # [batch_size, controller_state_size], for example.
            
            
            powers = self._powers # the powers of the rotation matrix the controller uses on rings
            powers21 = self._powers21 # the powers applied by ring 2 to ring 1

            h0, r1, w1, r2, w2, r3, w3, M1, M2, M3 = tf.split(state, [css, mas[0], mas[0], 
                                                        mas[1], mas[1], mas[2], mas[2],
                                                        mas[0] * mcs[0], mas[1] * mcs[1],
                                                        mas[2] * mcs[2]], 1)
            
            r = [r1,r2,r3]
            w = [w1,w2,w3]
            M = [M1,M2,M3]
            
            init = init_ops.constant_initializer(0.0)
            
            # Sharpening factor gamma, one for read and one for write, for each ring
            gamma_read_tensors = []
            gamma_write_tensors = []
            
            for i in range(num_rings):
                W_gamma_read = tf.get_variable("W_gamma_read" + str(i + 1), [css,1])
                B_gamma_read = tf.get_variable("B_gamma_read" + str(i + 1), [], initializer=init)
                gamma_read = 1.0 + tf.nn.relu(tf.matmul(h0,W_gamma_read) + B_gamma_read) # shape [batch_size,1]
                
                gamma_read_tensors.append(gamma_read)
                
                W_gamma_write = tf.get_variable("W_gamma_write" + str(i + 1), [css,1])
                B_gamma_write = tf.get_variable("B_gamma_write" + str(i + 1), [], initializer=init)
                gamma_write = 1.0 + tf.nn.relu(tf.matmul(h0,W_gamma_write) + B_gamma_write) # shape [batch_size,1]
                
                gamma_write_tensors.append(gamma_write)
                
            # Now generate the s, q, e, a vectors
            s_tensors = []
            q_tensors = []
            e_tensors = []
            a_tensors = []
            
            for i in range(num_rings):
                W_s = tf.get_variable("W_s" + str(i+1), [css,len(powers[i])])
                B_s = tf.get_variable("B_s" + str(i+1), [len(powers[i])], initializer=init)
                s = tf.nn.softmax(tf.matmul(h0,W_s) + B_s) # shape [batch_size,len(powers[i])]
                s_tensors.append(s)
                
                W_q = tf.get_variable("W_q" + str(i+1), [css,len(powers[i])])
                B_q = tf.get_variable("B_q" + str(i+1), [len(powers[i])], initializer=init)
                q = tf.nn.softmax(tf.matmul(h0,W_q) + B_q) # shape [batch_size,len(powers[i])]
                q_tensors.append(q)
                
                W_e = tf.get_variable("W_e" + str(i+1), [css,mcs[i]])
                B_e = tf.get_variable("B_e" + str(i+1), [mcs[i]], initializer=init)
                e = tf.sigmoid(tf.matmul(h0,W_e) + B_e) # shape [batch_size,mcs]
                e_tensors.append(e)
                
                W_a = tf.get_variable("W_a" + str(i+1), [css,mcs[i]])
                B_a = tf.get_variable("B_a" + str(i+1), [mcs[i]], initializer=init)
                a = tf.nn.relu(tf.matmul(h0,W_a) + B_a) # shape [batch_size,mcs]
                a_tensors.append(a)
                
            # DEBUG maybe add on ring 2 should be softmax?
            
            # Add and forget on the memory
            M_news = []
            
            for i in range(num_rings):
                M[i] = tf.reshape(M[i], [-1, mas[i], mcs[i]])
                erase_term = tf.matmul( M[i], tf.matrix_diag(e_tensors[i]) ) # shape [batch_size, mas[i], mcs[i]]
                add_term = tf.matmul( tf.reshape(w[i],[-1,mas[i],1]), tf.reshape(a_tensors[i],[-1,1,mcs[i]]) ) # shape [batch_size, mas[i], mcs[i]]
                M_new = M[i] - erase_term + add_term
                M_new = tf.reshape(M_new, [-1, mas[i] * mcs[i]])
                M_news.append(M_new)
                        
            # Do the rotations of the read and write addresses
            r_news = []
            w_news = []
            
            for i in range(num_rings):
                rot = rotation_tensor(mas[i],powers[i])
                
                w_new = tf.matmul( tf.reshape(w[i], [-1,1,mas[i]]),
                                tf.tensordot( s_tensors[i], rot, [[1], [0]] ) )
                r_new = tf.matmul( tf.reshape(r[i], [-1,1,mas[i]]),
                                tf.tensordot( q_tensors[i], rot, [[1], [0]] ) )
               
                r_new = tf.reshape( r_new, [-1,mas[i]] )
                w_new = tf.reshape( w_new, [-1,mas[i]] )
                r_news.append(r_new)
                w_news.append(w_new)
            
            # Special Pattern NTM stuff. We use the current contents of the
            # second memory ring to generate a distribution over rotations
            # of the first memory ring, and then act with those rotations on
            # the read address of the first memory ring. 
            
            rot = rotation_tensor(mas[0], powers21)
                
            Mr2 = tf.matmul( M[1], tf.reshape(r[1],[-1,mas[1],1]), transpose_a=True )
            Mr2 = tf.reshape( Mr2, [-1,mcs[1]] )           
            
            Mr3 = tf.matmul( M[2], tf.reshape(r[2],[-1,mas[2],1]), transpose_a=True ) 
            Mr3 = tf.reshape( Mr3, [-1,1] )
            
            # ASSUME mcs[1] = len(powers21)
            # We read the content of the third memory ring as a weight
            Mr2_rot = tf.tensordot( Mr2, rot, [[1], [0]] ) # shape [batch_size, mas[0], mas[0]]
            
            ident_mat = tf.diag(tf.ones([1,mas[0]], tf.float32)) # shape [mas[0], mas[0]]
            ident_mat = tf.reshape( ident_mat, [1, mas[0], mas[0]] )
            Mr3 = tf.tensordot( Mr3, ident_mat, [[1], [0]] ) # shape [batch_size, mas[0], mas[0]]
            
            rot_matrix = tf.matmul( Mr2_rot, Mr3 ) # shape [batch_size, mas[0], mas[0]]
            r_news[0] = tf.matmul( tf.reshape(r_news[0], [-1,1,mas[0]]), rot_matrix )
            r_news[0] = tf.reshape( r_news[0], [-1,mas[0]] )
            
            # Perform sharpening
            for i in range(num_rings):
                r_new = r_news[i]
                sharpening_tensor_r = tf.zeros_like(r_new) + gamma_read_tensors[i]
                sharp_r = tf.pow(r_new + 1e-6, sharpening_tensor_r)
                denom_r = tf.reduce_sum(sharp_r, axis=1, keep_dims=True)
                r_new = sharp_r / denom_r
                r_news[i] = r_new
                
                w_new = w_news[i]
                sharpening_tensor_w = tf.zeros_like(w_new) + gamma_write_tensors[i]
                sharp_w = tf.pow(w_new + 1e-6, sharpening_tensor_w)
                denom_w = tf.reduce_sum(sharp_w, axis=1, keep_dims=True)
                w_new = sharp_w / denom_w
                w_news[i] = w_new
                            
            # Now the usual RNN stuff
            H = tf.get_variable("H", [css,css])
            U = tf.get_variable("U", [self._input_size,css])
            B = tf.get_variable("B", [css], initializer=init)
        
            # Special Pattern NTM stuff, read from the first memory ring
            V = tf.get_variable("V", [mcs[0],css]) # converts from memory to controller state
            Mr1 = tf.matmul( M[0], tf.reshape(r[0],[-1,mas[0],1]), transpose_a=True )
            Mr1 = tf.reshape( Mr1, [-1,mcs[0]] )
            
            h0_new = tf.nn.tanh(tf.matmul(h0, H) + tf.matmul(Mr1,V) + tf.matmul(input,U) + B)
                    
            state_new = tf.concat([h0_new, r_news[0], w_news[0],
                                        r_news[1], w_news[1],
                                        r_news[2], w_news[2],
                                        M_news[0], M_news[1], M_news[2]], 1)   
        return h0_new, state_new
        # the return is output, state