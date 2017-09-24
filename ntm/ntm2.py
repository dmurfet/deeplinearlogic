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
from tensorflow.python.ops.math_ops import tanh

import ntm

##################
# Encoded Pattern NTM Controller
#
# This controller has two phases: when it is reading the pattern (which is
# the first part of the input tape) it is in Phase 1, while when it is reading
# the rest of the tape it is in Phase 2. The switch between these two phases
# happens when the controller sees the _second_ occurrence of trigger_symbol
# (which is set during initialisation and should be the encoded form of the
# initial symbol, in the current setup).
#
# At the end of Phase 1, the controller takes the accumulated inputs over the
# phase (so, the encoded form of the pattern) and uses a feed-forward network
# to generate from this the memory state of the second ring to the use for Phase 2.
#
# That is, we generate M2 and then in Phase 2 this is not modified. Otherwise it
# is identical the pattern NTM, except that we also do not allow any writing to 
# the second ring.
    
class EncodedPatternNTM(RNNCell):
    """
    The main Encoded Pattern NTM code.
    """
    def __init__(self, num_units, input_size, controller_state_size,
                memory_address_sizes,memory_content_sizes, 
                powers, powers21, direct_bias):
        self._num_units = num_units
        self._input_size = input_size
        self._controller_state_size = controller_state_size
        self._memory_address_sizes = memory_address_sizes
        self._memory_content_sizes = memory_content_sizes
        self._powers = powers
        self._powers21 = powers21
        self._direct_bias = direct_bias
        
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

            h0, r1, w1, r2, w2, M1, M2 = tf.split(state, [css, mas[0], mas[0], 
                                                        mas[1], mas[1],
                                                        mas[0] * mcs[0], mas[1] * mcs[1]], 1)
            
            r = [r1,r2]
            w = [w1,w2]
            M = [M1,M2]
            
            init = init_ops.constant_initializer(0.0)
            
            # Interpolation (note the nonzero initialisation for bias)
            W_interp = tf.get_variable("W_interp", [css,1])
            B_interp = tf.get_variable("B_interp", [1], initializer=init_ops.constant_initializer(self._direct_bias))
            interp = tf.nn.sigmoid(tf.matmul(h0,W_interp) + B_interp) # shape [batch_size,1]
                        
            # Sharpening factor gamma, one for read and one for write, for each ring
            gamma_read_tensors = []
            gamma_write_tensors = []
            
            for i in range(num_rings):
                W_gamma_read = tf.get_variable("W_gamma_read" + str(i + 1), [css,1])
                B_gamma_read = tf.get_variable("B_gamma_read" + str(i + 1), [], initializer=init)
                gamma_read = 1.0 + tf.nn.relu(tf.matmul(h0,W_gamma_read) + B_gamma_read) # shape [batch_size,1]
                
                gamma_read_tensors.append(gamma_read)
     
                # We only sharpen the write address of the first ring
                if( i == 0 ):           
                    W_gamma_write = tf.get_variable("W_gamma_write" + str(i + 1), [css,1])
                    B_gamma_write = tf.get_variable("B_gamma_write" + str(i + 1), [], initializer=init)
                    gamma_write = 1.0 + tf.nn.relu(tf.matmul(h0,W_gamma_write) + B_gamma_write) # shape [batch_size,1]
                
                    gamma_write_tensors.append(gamma_write)
                
            # Now generate the s, q, e, a vectors
            s_tensors = []
            q_tensors = []
            e_tensors = []
            a_tensors = []
            
            use_logits = [False, True]
            
            for i in range(num_rings):
            
                # We only need to rotate the the first write address
                if( i == 0 ):
                    W_s = tf.get_variable("W_s" + str(i+1), [css,len(powers[i])])
                    B_s = tf.get_variable("B_s" + str(i+1), [len(powers[i])], initializer=init)
                    s = tf.nn.softmax(tf.matmul(h0,W_s) + B_s) # shape [batch_size,len(powers[i])]
                    s_tensors.append(s)
                
                W_q = tf.get_variable("W_q" + str(i+1), [css,len(powers[i])])
                B_q = tf.get_variable("B_q" + str(i+1), [len(powers[i])], initializer=init)
                q = tf.nn.softmax(tf.matmul(h0,W_q) + B_q) # shape [batch_size,len(powers[i])]
                q_tensors.append(q)
                
                # We only write to the first memory ring
                if( i == 0 ):
                    W_e = tf.get_variable("W_e" + str(i+1), [css,mcs[i]])
                    B_e = tf.get_variable("B_e" + str(i+1), [mcs[i]], initializer=init)
                    e = tf.nn.sigmoid(tf.matmul(h0,W_e) + B_e) # shape [batch_size,mcs]
                    e_tensors.append(e)
                
                    W_a = tf.get_variable("W_a" + str(i+1), [css,mcs[i]])
                    B_a = tf.get_variable("B_a" + str(i+1), [mcs[i]], initializer=init)
                
                    if( use_logits[i] == False ):
                        a = tf.nn.relu(tf.matmul(h0,W_a) + B_a) # shape [batch_size,mcs]
                    else:
                        a = tf.matmul(h0,W_a) + B_a # shape [batch_size,mcs]
                    a_tensors.append(a)
                                    
            # Do the rotations of the read and write addresses
            r_news = []
            w_news = []
            
            for i in range(num_rings):
                rot = rotation_tensor(mas[i],powers[i])
                
                # For the second memory ring we just leave w constant
                if( i == 0 ):
                    w_new = tf.matmul( tf.reshape(w[i], [-1,1,mas[i]]),
                                    tf.tensordot( s_tensors[i], rot, [[1], [0]] ) )
                    w_new = tf.reshape( w_new, [-1,mas[i]] )
                    w_news.append(w_new)
                else:
                    w_news.append(w[i])
                
                r_new = tf.matmul( tf.reshape(r[i], [-1,1,mas[i]]),
                                tf.tensordot( q_tensors[i], rot, [[1], [0]] ) )
                r_new = tf.reshape( r_new, [-1,mas[i]] )
                r_news.append(r_new)

            for i in range(num_rings):
                M[i] = tf.reshape(M[i], [-1, mas[i], mcs[i]])
 
            # Special Pattern NTM stuff. We use the current contents of the
            # second memory ring to generate a distribution over rotations
            # of the first memory ring, and then act with those rotations on
            # the read address of the first memory ring. 
            
            rot = rotation_tensor(mas[0], powers21)
                
            Mr2 = tf.matmul( tf.nn.softmax(M[1]), tf.reshape(r[1],[-1,mas[1],1]), transpose_a=True )
            Mr2 = tf.reshape( Mr2, [-1,mcs[1]] )
                        
            # ASSUME mcs[1] = len(powers21)
            Mr2_rot = tf.tensordot( Mr2, rot, [[1], [0]] ) # shape [batch_size, mas[0], mas[0]]
            r0_prime = tf.matmul( tf.reshape(r[0], [-1,1,mas[0]]), Mr2_rot )
            r0_prime = tf.reshape( r0_prime, [-1,mas[0]] )
            
            # We view the scalar interp as a weight interpolating
            # between the direct manipulation of the read address of the first ring
            # by the controller (that is, r_news[0]) and the indirect manipulation
            # via the contents of the second ring
            
            # For the version with interpolation based on contents of a memory
            # ring see ntm-2-4-2017-snapshot
            
            interp_matrix = tf.stack([interp,
                                    tf.ones_like(interp,dtype=tf.float32) - interp],
                                    axis=1) # shape [-1,2,1]
            
            r0_new = tf.matmul( tf.stack([r_news[0], r0_prime], axis=2), interp_matrix )
            #                   [---- this is shape [-1,mas[0],2] ----] 
            
            r0_new = tf.reshape( r0_new, [-1, mas[0]] )
            r_news[0] = r0_new

            # Perform sharpening (for the second ring we only sharpen the read address
            # as the encoded pattern NTM does not write to the second ring and therefore
            # does not use the write address
            for i in range(num_rings):
                r_new = r_news[i]
                sharpening_tensor_r = tf.zeros_like(r_new) + gamma_read_tensors[i]
                sharp_r = tf.pow(r_new + 1e-6, sharpening_tensor_r)
                denom_r = tf.reduce_sum(sharp_r, axis=1, keep_dims=True)
                r_new = sharp_r / denom_r
                r_news[i] = r_new
                
                if( i == 0 ):
                    w_new = w_news[i]
                    sharpening_tensor_w = tf.zeros_like(w_new) + gamma_write_tensors[i]
                    sharp_w = tf.pow(w_new + 1e-6, sharpening_tensor_w)
                    denom_w = tf.reduce_sum(sharp_w, axis=1, keep_dims=True)
                    w_new = sharp_w / denom_w
                    w_news[i] = w_new
 
            # Add and forget on the memory
            M_news = []
            
            # Add and erase on memory (first ring only)
            for i in range(1):
                erase_term1 = tf.matmul( tf.reshape(w_news[i],[-1,mas[i],1]), tf.reshape(e_tensors[i],[-1,1,mcs[i]]) ) # shape [batch_size, mas[i], mcs[i]]
                erase_term = tf.multiply( M[i], erase_term1 ) # shape [batch_size, mas[i], mcs[i]]
                add_term = tf.matmul( tf.reshape(w_news[i],[-1,mas[i],1]), tf.reshape(a_tensors[i],[-1,1,mcs[i]]) ) # shape [batch_size, mas[i], mcs[i]]
                M_new = M[i] - erase_term + add_term
                M_new = tf.reshape(M_new, [-1, mas[i] * mcs[i]])
                M_news.append(M_new)
                
            # The memory state of the second memory ring stays constant
            M_new = tf.reshape(M[1], [-1, mas[1] * mcs[1]])
            M_news.append(M_new)
                           
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
                                        M_news[0], M_news[1]], 1)   
        return h0_new, state_new
        # the return is output, state