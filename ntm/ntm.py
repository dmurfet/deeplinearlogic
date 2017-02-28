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
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

# GLOBAL FLAGS
batch_size = 1000
controller_state_size = 10 # dimension of the internal state space of the controller
memory_address_size = 5 # number of memory locations
memory_content_size = 5 # size of vector stored at a memory location
state_size = controller_state_size + 2*memory_address_size \
			+ memory_address_size * memory_content_size

class NTMRNNCell(RNNCell):
    
    def __init__(self, num_units, input_size=None, activation=tanh):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated." % self)
        self._num_units = num_units
        self._activation = activation
    
    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        # the scope business gives a namespace to our weight variable matrix names
        with vs.variable_scope(scope or "NTM"): 
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
            h0, r, w, M = tf.split(state, [controller_state_size,
            								memory_address_size,
            								memory_address_size,
            								memory_address_size * memory_content_size], 1)
            								
            H = vs.get_variable("H", [controller_state_size,controller_state_size])
            h0_n = tf.matmul(h0, H)
            
            output = tf.concat([h0_n,r,w,M], 1)   
        return output, output
        # note that as currently written the RNN emits its internal state at each time step