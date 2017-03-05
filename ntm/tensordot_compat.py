from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import state_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_math_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated

def tensordot(a, b, axes, name=None):
  r"""Tensor contraction of a and b along specified axes.

  Tensordot (also known as tensor contraction) sums the product of elements
  from `a` and `b` over the indices specified by `a_axes` and `b_axes`.
  The lists `a_axes` and `b_axes` specify those pairs of axes along which to
  contract the tensors. The axis `a_axes[i]` of `a` must have the same dimension
  as axis `b_axes[i]` of `b` for all `i` in `range(0, len(a_axes))`. The lists
  `a_axes` and `b_axes` must have identical length and consist of unique
  integers that specify valid axes for each of the tensors.

  This operation corresponds to `numpy.tensordot(a, b, axes)`.

  Example 1: When `a` and `b` are matrices (order 2), the case `axes = 1`
  is equivalent to matrix multiplication.

  Example 2: When `a` and `b` are matrices (order 2), the case
  `axes = [[1], [0]]` is equivalent to matrix multiplication.

  Example 3: Suppose that \\(a_ijk\\) and \\(b_lmn\\) represent two
  tensors of order 3. Then, `contract(a, b, [0], [2])` is the order 4 tensor
  \\(c_{jklm}\\) whose entry
  corresponding to the indices \\((j,k,l,m)\\) is given by:

  \\( c_{jklm} = \sum_i a_{ijk} b_{lmi} \\).

  In general, `order(c) = order(a) + order(b) - 2*len(axes[0])`.

  Args:
    a: `Tensor` of type `float32` or `float64`.
    b: `Tensor` with the same type as `a`.
    axes: Either a scalar `N`, or a list or an `int32` `Tensor` of shape [2, k].
     If axes is a scalar, sum over the last N axes of a and the first N axes
     of b in order.
     If axes is a list or `Tensor` the first and second row contain the set of
     unique integers specifying axes along which the contraction is computed,
     for `a` and `b`, respectively. The number of axes for `a` and `b` must
     be equal.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as `a`.

  Raises:
    ValueError: If the shapes of `a`, `b`, and `axes` are incompatible.
    IndexError: If the values in axes exceed the rank of the corresponding
      tensor.
  """

  def _tensordot_reshape(a, axes, flipped=False):
    """Helper method to perform transpose and reshape for contraction op.

    This method is helpful in reducing `math_ops.tensordot` to `math_ops.matmul`
    using `array_ops.transpose` and `array_ops.reshape`. The method takes a
    tensor and performs the correct transpose and reshape operation for a given
    set of indices. It returns the reshaped tensor as well as a list of indices
    necesary to reshape the tensor again after matrix multiplication.

    Args:
      a: `Tensor`.
      axes: List or `int32` `Tensor` of unique indices specifying valid axes of
       `a`.
      flipped: An optional `bool`. Defaults to `False`. If `True`, the method
        assumes that `a` is the second argument in the contraction operation.

    Returns:
      A pair `(reshaped_a, free_dims)` where `reshaped_a` is the tensor `a`
      reshaped to allow contraction via `matmul` and `free_dims` is either a
      list of integers or an `int32` `Tensor`, depending on if `axes` is a list
      and the shape of `a`  is fully defined.
    """
    # TODO(b/33084409): Implement partial shape inference.
    if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
      shape_a = a.get_shape().as_list()
      axes = [i if i >= 0 else i + len(shape_a) for i in axes]
      free = [i for i in xrange(len(shape_a)) if i not in axes]
      free_dims = [shape_a[i] for i in free]
      prod_free = int(np.prod([shape_a[i] for i in free]))
      prod_axes = int(np.prod([shape_a[i] for i in axes]))
      perm = list(axes) + free if flipped else free + list(axes)
      new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
      reshaped_a = array_ops.reshape(array_ops.transpose(a, perm), new_shape)
      return reshaped_a, free_dims
    else:
      shape_a = array_ops.shape(a)
      rank_a = array_ops.rank(a)
      axes = ops.convert_to_tensor(axes, dtype=dtypes.int32, name="axes")
      axes = cast(axes >= 0, dtypes.int32) * axes + cast(
          axes < 0, dtypes.int32) * (axes + rank_a)
      free, _ = array_ops.setdiff1d(range(rank_a), axes)
      free_dims = array_ops.gather(shape_a, free)
      axes_dims = array_ops.gather(shape_a, axes)
      prod_free_dims = reduce_prod(free_dims)
      prod_axes_dims = reduce_prod(axes_dims)
      perm = array_ops.concat([axes_dims, free_dims], 0)
      if flipped:
        perm = array_ops.concat([axes, free], 0)
        new_shape = array_ops.stack([prod_axes_dims, prod_free_dims])
      else:
        perm = array_ops.concat([free, axes], 0)
        new_shape = array_ops.stack([prod_free_dims, prod_axes_dims])
      reshaped_a = array_ops.reshape(array_ops.transpose(a, perm), new_shape)
      return reshaped_a, free_dims

  def _tensordot_axes(a, axes):
    """Generates two sets of contraction axes for the two tensor arguments."""
    a_shape = a.get_shape()
    if isinstance(axes, compat.integral_types):
      if axes < 1:
        raise ValueError("'axes' must be at least 1.")
      if a_shape.ndims is not None:
        return range(a_shape.ndims - axes, a_shape.ndims), range(axes)
      else:
        rank = array_ops.rank(a)
        return (array_ops.range(
            rank - axes, rank, dtype=dtypes.int32), array_ops.range(
                rank, dtype=dtypes.int32))
    elif isinstance(axes, (list, tuple)):
      if len(axes) != 2:
        raise ValueError("'axes' must be an integer or have length 2.")
      a_axes = axes[0]
      b_axes = axes[1]
      if len(a_axes) != len(b_axes):
        raise ValueError(
            "Different number of contraction axes 'a' and 'b', %s != %s.",
            len(a_axes), len(b_axes))
      return a_axes, b_axes
    else:
      axes = ops.convert_to_tensor(axes, name="axes", dtype=dtypes.int32)
      return axes[0], axes[1]

  with ops.name_scope(name, "Tensordot", [a, b, axes]) as name:
    a = ops.convert_to_tensor(a, name="a")
    b = ops.convert_to_tensor(b, name="b")
    a_axes, b_axes = _tensordot_axes(a, axes)
    a_reshape, a_free_dims = _tensordot_reshape(a, a_axes)
    b_reshape, b_free_dims = _tensordot_reshape(b, b_axes, True)
    ab_matmul = tf.matmul(a_reshape, b_reshape)
    if isinstance(a_free_dims, list) and isinstance(b_free_dims, list):
      return array_ops.reshape(ab_matmul, a_free_dims + b_free_dims, name=name)
    else:
      a_free_dims = ops.convert_to_tensor(a_free_dims)
      b_free_dims = ops.convert_to_tensor(b_free_dims)
      return array_ops.reshape(
          ab_matmul, array_ops.concat([a_free_dims, b_free_dims], 0), name=name)