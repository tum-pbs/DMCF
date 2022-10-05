# pointnet-autoencoder

# Copyright (c) 2018, Geometric Computation Group of Stanford University

# The MIT License (MIT)

# Copyright (c) 2018 Charles R. Qi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops
import os.path as osp

base_dir = osp.dirname(osp.abspath(__file__))

if osp.exists(osp.join(base_dir, 'nn_distance_so.so')):
    nn_distance_module = tf.load_op_library(
        osp.join(base_dir, 'nn_distance_so.so'))
else:
    print("WARNING: nn_distance module not installed!")


def nn_distance(xyz1, xyz2):
    '''
	Computes the distance of nearest neighbors for a pair of point clouds
	input: xyz1: (batch_size,#points_1,3)  the first point cloud
	input: xyz2: (batch_size,#points_2,3)  the second point cloud
	output: dist1: (batch_size,#point_1)   distance from first to second
	output: idx1:  (batch_size,#point_1)   nearest neighbor from first to second
	output: dist2: (batch_size,#point_2)   distance from second to first
	output: idx2:  (batch_size,#point_2)   nearest neighbor from second to first
	'''

    return nn_distance_module.nn_distance(xyz1, xyz2)


#@ops.RegisterShape('NnDistance')
#def _nn_distance_shape(op):
#	shape1=op.inputs[0].get_shape().with_rank(3)
#	shape2=op.inputs[1].get_shape().with_rank(3)
#	return [tf.TensorShape([shape1.dims[0],shape1.dims[1]]),tf.TensorShape([shape1.dims[0],shape1.dims[1]]),
#		tf.TensorShape([shape2.dims[0],shape2.dims[1]]),tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
@tf.RegisterGradient('NnDistance')
def _nn_distance_grad(op, grad_dist1, grad_idx1, grad_dist2, grad_idx2):
    xyz1 = op.inputs[0]
    xyz2 = op.inputs[1]
    idx1 = op.outputs[1]
    idx2 = op.outputs[3]
    return nn_distance_module.nn_distance_grad(xyz1, xyz2, grad_dist1, idx1,
                                               grad_dist2, idx2)


def batch_gather(params, indices, name=None):
    """Gather slices from `params` according to `indices` with leading batch dims.
  This operation assumes that the leading dimensions of `indices` are dense,
  and the gathers on the axis corresponding to the last dimension of `indices`.
  More concretely it computes:
  result[i1, ..., in] = params[i1, ..., in-1, indices[i1, ..., in]]
  Therefore `params` should be a Tensor of shape [A1, ..., AN, B1, ..., BM],
  `indices` should be a Tensor of shape [A1, ..., AN-1, C] and `result` will be
  a Tensor of size `[A1, ..., AN-1, C, B1, ..., BM]`.
  In the case in which indices is a 1D tensor, this operation is equivalent to
  `tf.gather`.
  See also `tf.gather` and `tf.gather_nd`.
  Args:
    params: A Tensor. The tensor from which to gather values.
    indices: A Tensor. Must be one of the following types: int32, int64. Index
        tensor. Must be in range `[0, params.shape[axis]`, where `axis` is the
        last dimension of `indices` itself.
    name: A name for the operation (optional).
  Returns:
    A Tensor. Has the same type as `params`.
  Raises:
    ValueError: if `indices` has an unknown shape.
  """

    with ops.name_scope(name):
        indices = ops.convert_to_tensor(indices, name="indices")
        params = ops.convert_to_tensor(params, name="params")
        indices_shape = tf.shape(indices)
        params_shape = tf.shape(params)
        ndims = indices.shape.ndims
        if ndims is None:
            raise ValueError(
                "batch_gather does not allow indices with unknown "
                "shape.")
        batch_indices = indices
        accum_dim_value = 1
        for dim in range(ndims - 1, 0, -1):
            dim_value = params_shape[dim - 1]
            accum_dim_value *= params_shape[dim]
            dim_indices = gen_math_ops._range(0, dim_value, 1)
            dim_indices *= accum_dim_value
            dim_shape = tf.stack([1] * (dim - 1) + [dim_value] + [1] *
                                 (ndims - dim),
                                 axis=0)
            batch_indices += tf.cast(tf.reshape(dim_indices, dim_shape),
                                     tf.int64)

        flat_indices = tf.reshape(batch_indices, [-1])
        outer_shape = params_shape[ndims:]
        flat_inner_shape = gen_math_ops.prod(params_shape[:ndims], [0], False)

        flat_params = tf.reshape(
            params, tf.concat([[flat_inner_shape], outer_shape], axis=0))
        flat_result = tf.gather(flat_params, flat_indices)
        result = tf.reshape(flat_result,
                            tf.concat([indices_shape, outer_shape], axis=0))
        final_shape = indices.get_shape()[:ndims - 1].merge_with(
            params.get_shape()[:ndims - 1])
        final_shape = final_shape.concatenate(indices.get_shape()[ndims - 1])
        final_shape = final_shape.concatenate(params.get_shape()[ndims:])
        result.set_shape(final_shape)
        return result


def nn_index(y_true, y_pred):
    distance = tf.reduce_sum(
        tf.square(tf.expand_dims(y_true, 2) - tf.expand_dims(y_pred, 1)), -1)
    return tf.argmin(distance, -1)


def chamfer_loss(y_true, y_pred):
    if len(y_true.shape) == 2:
        y_true = tf.expand_dims(y_true, axis=0)
    if len(y_pred.shape) == 2:
        y_pred = tf.expand_dims(y_pred, axis=0)
    cost_p1_p2, _, cost_p2_p1, _ = nn_distance(y_pred, y_true)
    return tf.reduce_mean(cost_p1_p2, axis=-1) + tf.reduce_mean(cost_p2_p1,
                                                                axis=-1)
