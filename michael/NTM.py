"""
This module implements a neural turing machine.
Not finished.
Still need to debug code. 
"""
import math
import autograd.numpy as np
from autograd import grad, jacobian
from util.util import rando, sigmoid, softmax, softplus, unwrap, sigmoid_prime, tanh_prime, compare_deltas, dKdu, softmax_grads, beta_grads, K_focus
import mastermemory
import addressingmechanisms
from addressingmechanisms import cosine_sim, shift
import sys

class NTM(object):
  """
  NTM with a single-layer feed-forward controller, using autodiff
  """

  def __init__(self, in_size, out_size, hidden_size, N, M, vec_size):

    self.N = N  # the number of memory locations
    self.M = M # the number of columns in a memory location
    self.out_size = out_size
    self.vec_size = vec_size
    shift_width = min(3,self.N) # seems necessary for generalization

    self.stats = None

    self.W = {} # maps parameter names to tensors

    # non-head parameters
    self.W['xh'] = rando(hidden_size, in_size)
    self.W['ho'] = rando(hidden_size, hidden_size)
    self.W['oy'] = rando(out_size, hidden_size)
    self.W['bh']  = rando(hidden_size, 1)
    self.W['by']  = rando(out_size, 1)
    self.W['bo']  = rando(hidden_size, 1)

    # weights from last read head output to hidden layer
    self.W['rh'] = rando(hidden_size, self.M)

    # weights
    self.W['ok_r'] = rando(self.M,hidden_size)
    self.W['ok_w'] = rando(self.M,hidden_size)

    self.W['obeta_r'] = rando(1,hidden_size)
    self.W['obeta_w'] = rando(1,hidden_size)

    # the interpolation gate is a scalar
    self.W['og_r'] = rando(1,hidden_size)
    self.W['og_w'] = rando(1,hidden_size)

    self.W['os_r'] = rando(shift_width,hidden_size)
    self.W['os_w'] = rando(shift_width,hidden_size)

    # gamma is also a scalar
    self.W['ogamma_r'] = rando(1,hidden_size)
    self.W['ogamma_w'] = rando(1,hidden_size)

    self.W['oadds']   = rando(self.M,hidden_size)
    self.W['oerases'] = rando(self.M,hidden_size)

    # biases
    self.W['bk_r'] = rando(self.M,1)
    self.W['bk_w'] = rando(self.M,1)

    self.W['bbeta_r'] = rando(1,1)
    self.W['bbeta_w'] = rando(1,1)

    self.W['bg_r'] = rando(1,1)
    self.W['bg_w'] = rando(1,1)

    self.W['bs_r'] = rando(shift_width,1)
    self.W['bs_w'] = rando(shift_width,1)

    self.W['bgamma_r'] = rando(1,1)
    self.W['bgamma_w'] = rando(1,1)

    self.W['badds']  = rando(self.M,1)
    self.W['berases'] = rando(self.M,1)

    # parameters specifying initial conditions
    self.W['rsInit'] = np.random.uniform(-1,1,(self.M,1))
    self.W['w_wsInit'] = np.random.randn(self.N,1)*0.01
    self.W['w_rsInit'] = np.random.randn(self.N,1)*0.01

    # initial condition of the memory
    self.W['memsInit'] = np.random.randn(self.N,self.M)*0.01

  def lossFunction(self, inputs, targets, manual_grad=False):
    """
    Returns the loss given an inputs,targets tuple
    """

    def fprop(params):
      """
      Forward pass of the NTM.
      """

      W = params # aliasing for brevity

      xs, zhs, hs, ys, ps, ts, zos, os = {}, {}, {}, {}, {}, {}, {}, {}

      def l():
        """
        Silly utility function that should be called in init.
        """
        return {}

      rs = l()
      zk_rs = l() 
      k_rs, beta_rs, g_rs, s_rs, gamma_rs = l(),l(),l(),l(),l()
      k_ws, beta_ws, g_ws, s_ws, gamma_ws = l(),l(),l(),l(),l()
      adds, erases = l(),l()
      zbeta_rs, zbeta_ws = l(),l()
      zs_rs, zs_ws = l(),l()
      wg_rs, wg_ws = l(),l()
      w_ws, w_rs = l(),l() # read weights and write weights
      wc_ws, wc_rs = l(),l() # read and write content weights
      rs[-1] = self.W['rsInit'] # stores values read from memory
      w_ws[-1] = softmax(self.W['w_wsInit'])
      w_rs[-1] = softmax(self.W['w_rsInit'])

      mems = {} # the state of the memory at every timestep
      mems[-1] = self.W['memsInit']
      loss = 0

      for t in xrange(len(inputs)):

        xs[t] = np.reshape(np.array(inputs[t]),inputs[t].shape[::-1])

        rsum = np.dot(W['rh'], np.reshape(rs[t-1],(self.M,1)))
        zhs[t] = np.dot(W['xh'], xs[t]) + rsum + W['bh']
        hs[t] = np.tanh(zhs[t])

        zos[t] = np.dot(W['ho'], hs[t]) + W['bo']
        os[t] = np.tanh(zos[t])

        # parameters to the read head
        zk_rs[t] =np.dot(W['ok_r'],os[t]) + W['bk_r']
        k_rs[t] = np.tanh(zk_rs[t])
        zbeta_rs[t] = np.dot(W['obeta_r'],os[t]) + W['bbeta_r']
        beta_rs[t] = softplus(zbeta_rs[t])
        g_rs[t] = sigmoid(np.dot(W['og_r'],os[t]) + W['bg_r'])
        zs_rs[t] = np.dot(W['os_r'],os[t]) + W['bs_r']
        s_rs[t] = softmax(zs_rs[t])
        gamma_rs[t] = 1 + sigmoid(np.dot(W['ogamma_r'], os[t])
                                        + W['bgamma_r'])

        # parameters to the write head
        k_ws[t] = np.tanh(np.dot(W['ok_w'],os[t]) + W['bk_w'])
        zbeta_ws[t] = np.dot(W['obeta_w'],os[t]) + W['bbeta_w']
        beta_ws[t] = softplus(zbeta_ws[t])
        g_ws[t] = sigmoid(np.dot(W['og_w'],os[t]) + W['bg_w'])
        zs_ws[t] = np.dot(W['os_w'],os[t]) + W['bs_w']
        s_ws[t] = softmax(zs_ws[t])
        gamma_ws[t] = 1 + sigmoid(np.dot(W['ogamma_w'], os[t])
                                        + W['bgamma_w'])

        # the erase and add vectors
        # these are also parameters to the write head
        # but they describe "what" is to be written rather than "where"
        adds[t] = np.tanh(np.dot(W['oadds'], os[t]) + W['badds'])
        erases[t] = sigmoid(np.dot(W['oerases'], os[t]) + W['berases'])

        w_ws[t], wg_ws[t], wc_ws[t] = addressing.create_weights(   k_ws[t]
                                                , beta_ws[t]
                                                , g_ws[t]
                                                , s_ws[t]
                                                , gamma_ws[t]
                                                , w_ws[t-1]
                                                , mems[t-1])

        w_rs[t], wg_rs[t], wc_rs[t] = addressing.create_weights(   k_rs[t]
                                                , beta_rs[t]
                                                , g_rs[t]
                                                , s_rs[t]
                                                , gamma_rs[t]
                                                , w_rs[t-1]
                                                , mems[t-1])

        ys[t] = np.dot(W['oy'], os[t]) + W['by']
        ps[t] = sigmoid(ys[t])

        one = np.ones(ps[t].shape)
        ts[t] = np.reshape(np.array(targets[t]),(self.out_size,1))

        epsilon = 2**-23 # to prevent log(0)
        a = np.multiply(ts[t] , np.log(ps[t] + epsilon))
        b = np.multiply(one - ts[t], np.log(one-ps[t] + epsilon))
        loss = loss - (a + b)

        # read from the memory
        rs[t] = memory.read(mems[t-1],w_rs[t])

        # write into the memory
        mems[t] = memory.write(mems[t-1],w_ws[t],erases[t],adds[t])

      self.stats = [loss, mems, ps, ys, os, zos, hs, zhs, xs, rs, w_rs,
                    w_ws, adds, erases, k_rs, k_ws, g_rs, g_ws, wc_rs, wc_ws,
                    zbeta_rs, zbeta_ws, zs_rs, zs_ws, wg_rs, wg_ws]
      return np.sum(loss)

# review above (debugue) and continue... next. Think about it!
# def manual_grads(params)
# AIM:  return loss, deltas, ps, w_rs, w_ws, adds, erases