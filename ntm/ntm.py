"""
This module implements the (location addressing only) Neural Turing Machine in TensorFlow
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
# Memory

def memory_operation(h0, r, w, M):
	return r, w, M

##################
# NTM Controller
#
	
class NTMRNNCell(RNNCell):
    
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
            # inputs has shape [batch_size, input_size]
            # state has shape [batch_size, state_size]
            
            # We divide up a state vector as follows:
            #
            # h = (h0,r,w,M)
            #
            # where
            #
            #	- h0 is the controller internal state (size controller_state_size)
            #	- r is the read address weights (size memory_address_size)
            #	- w is the write address weights (size memory_address_size)
            #	- M is the memory state (size memory_address_size*memory_content_size)
			#
			# the memory state vector M is the concatenation of the vectors at each
			# memory location, in order. For example, if N = memory_address_size then
			#
			# M = M[0], M[1], ..., M[N-1]
			#
			# NOTE: these vectors are all batched, so in the following h0 has shape
			# [batch_size, controller_state_size], for example.
            
            # The first step is to decompose the state vector
            h0, r, w, M = tf.split(state, [self._controller_state_size,
            								self._memory_address_size,
            								self._memory_address_size,
            								self._memory_address_size * self._memory_content_size], 1)
            
            r_n, w_n, M_n = memory_operation(h0, r, w, M)
            			
            H = tf.get_variable("H", [self._controller_state_size,self._controller_state_size])
            U = tf.get_variable("U", [self._input_size,self._controller_state_size])
            B = tf.get_variable("B", [self._controller_state_size], initializer=init_ops.constant_initializer(0.0))
            
            h0_n = self._activation(tf.matmul(h0, H) + tf.matmul(input,U) + B)
            
            output = tf.concat([h0_n,r_n,w_n,M_n], 1)   
        return output, output
        # note that as currently written the RNN emits its internal state at each time step