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
import os.path as osp

base_dir = osp.dirname(osp.abspath(__file__))

if osp.exists(osp.join(base_dir, 'tf_approxmatch_so.so')):
    approxmatch_module = tf.load_op_library(
        osp.join(base_dir, 'tf_approxmatch_so.so'))
else:
    print("WARNING: tf_approxmatch_so module not installed!")


def approx_match(xyz1, xyz2, n=None, m=None):
    '''
input:
	xyz1 : batch_size * #dataset_points * 3
	xyz2 : batch_size * #query_points * 3
	n : batch_size * 1
	m : batch_size * 1
returns:
	match : batch_size * #query_points * #dataset_points
	'''
    if n is None:
        n = tf.tile(tf.shape(xyz1)[1:2], tf.shape(xyz1)[:1])
    if m is None:
        m = tf.tile(tf.shape(xyz2)[1:2], tf.shape(xyz2)[:1])
    return approxmatch_module.approx_match(xyz1, xyz2, n, m)


ops.NoGradient('ApproxMatch')


def match_cost(xyz1, xyz2, match):
    '''
input:
	xyz1 : batch_size * #dataset_points * 3
	xyz2 : batch_size * #query_points * 3
	match : batch_size * #query_points * #dataset_points
returns:
	cost : batch_size
	'''
    return approxmatch_module.match_cost(xyz1, xyz2, match)
