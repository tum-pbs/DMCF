import tensorflow as tf
import os

import functools
import numpy as np
import json
import tree

INPUT_SEQUENCE_LENGTH = 6

# Create a description of the features.
_FEATURE_DESCRIPTION = {
    'position': tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = tf.io.VarLenFeature(
    tf.string)

_FEATURE_DTYPES = {
    'position': {
        'in': np.float32,
        'out': tf.float32
    },
    'step_context': {
        'in': np.float32,
        'out': tf.float32
    }
}

_CONTEXT_FEATURES = {
    'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'particle_type': tf.io.VarLenFeature(tf.string)
}


def convert_to_tensor(x, encoded_dtype):
    if len(x) == 1:
        out = np.frombuffer(x[0].numpy(), dtype=encoded_dtype)
    else:
        out = []
        for el in x:
            out.append(np.frombuffer(el.numpy(), dtype=encoded_dtype))
    out = tf.convert_to_tensor(np.array(out))
    return out


def parse_serialized_simulation_example(example_proto, metadata):
    """Parses a serialized simulation tf.SequenceExample.

  Args:
    example_proto: A string encoding of the tf.SequenceExample proto.
    metadata: A dict of metadata for the dataset.

  Returns:
    context: A dict, with features that do not vary over the trajectory.
    parsed_features: A dict of tf.Tensors representing the parsed examples
      across time, where axis zero is the time axis.

  """
    if 'context_mean' in metadata:
        feature_description = _FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
    else:
        feature_description = _FEATURE_DESCRIPTION
    context, parsed_features = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=_CONTEXT_FEATURES,
        sequence_features=feature_description)
    for feature_key, item in parsed_features.items():
        convert_fn = functools.partial(
            convert_to_tensor,
            encoded_dtype=_FEATURE_DTYPES[feature_key]['in'])
        parsed_features[feature_key] = tf.py_function(
            convert_fn,
            inp=[item.values],
            Tout=_FEATURE_DTYPES[feature_key]['out'])

    # There is an extra frame at the beginning so we can calculate pos change
    # for all frames used in the paper.
    position_shape = [metadata['sequence_length'] + 1, -1, metadata['dim']]

    # Reshape positions to correct dim:
    parsed_features['position'] = tf.reshape(parsed_features['position'],
                                             position_shape)
    # Set correct shapes of the remaining tensors.
    sequence_length = metadata['sequence_length'] + 1
    if 'context_mean' in metadata:
        context_feat_len = len(metadata['context_mean'])
        parsed_features['step_context'] = tf.reshape(
            parsed_features['step_context'],
            [sequence_length, context_feat_len])
    # Decode particle type explicitly
    context['particle_type'] = tf.py_function(
        functools.partial(convert_fn, encoded_dtype=np.int64),
        inp=[context['particle_type'].values],
        Tout=[tf.int64])
    context['particle_type'] = tf.reshape(context['particle_type'], [-1])
    return context, parsed_features


def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())


class ParticleIdxGrid:

    def __init__(self, particles, shape):
        self.particles = particles
        self.shape = shape
        self.grid = np.empty(shape[::-1], dtype=object)
        self.grid[:] = [[[[] for x in range(shape[0])]
                         for y in range(shape[1])] for z in range(shape[2])]

        for i in range(len(particles)):
            x, y, z = particles[i].astype(dtype="int32")
            if x >= 0 and x < self.shape[0] and y >= 0 and y < self.shape[
                    1] and z >= 0 and z < self.shape[2]:
                self.grid[z, y, x].append(i)

    def get_cell(self, cell_idx):
        x, y, z = cell_idx.astype(dtype="int32")
        return self.grid[z, y, x]

    def get_range(self, c, r):
        sx, sy, sz = self.shape
        x0, y0, z0 = np.clip((c - r).astype('int32'), 0, [sx, sy, sz])
        x1, y1, z1 = np.clip((c + r).astype('int32'), 0, [sx, sy, sz])

        return [
            v for z in self.grid[z0:z1, y0:y1, x0:x1] for y in z for x in y
            for v in x
        ]

    def get_normal(self, pos, h=0.5):
        nn = self.get_range(pos, h * 3)
        if len(nn) < 2:
            return np.array([0, 0, 0])

        nn = self.particles[nn] - pos
        normal = -np.sum(
            nn * (np.exp(-np.sum(nn**2, axis=-1, keepdims=True) / h**2)),
            axis=0)
        l = np.linalg.norm(normal)
        if l > 1e-10:
            return normal / l
        return np.array([0, 0, 0])


def prepare_rollout_inputs(context, features):
    """Prepares an inputs trajectory for rollout."""
    # [sequence_length, num_particles, dim]
    out_dict = {
        'pos': features['position'],
        'type': context['particle_type']
    }
    if 'step_context' in features:
        out_dict['ctx'] = features['step_context']
    return out_dict


def get_input_fn(data_path, batch_size, mode, split):
    """Gets the learning simulation input function for tf.estimator.Estimator.

  Args:
    data_path: the path to the dataset directory.
    batch_size: the number of graphs in a batch.
    mode: either 'one_step_train', 'one_step' or 'rollout'
    split: either 'train', 'valid' or 'test.

  Returns:
    The input function for the learning simulation model.
  """

    def input_fn():
        """Input function for learning simulation."""
        # Loads the metadata of the dataset.
        metadata = _read_metadata(data_path)
        # Create a tf.data.Dataset from the TFRecord.
        ds = tf.data.TFRecordDataset(
            [os.path.join(data_path, f'{split}.tfrecord')])
        ds = ds.map(
            functools.partial(parse_serialized_simulation_example,
                              metadata=metadata))

        ds = ds.map(prepare_rollout_inputs)
        return ds

    return input_fn


def sample_box(fromX, toX, fromY, toY, fromZ, toZ):
    x_r, y_r, z_r = np.meshgrid(np.arange(fromX, toX),
                                np.arange(fromY, toY),
                                np.arange(fromZ, toZ),
                                indexing='ij')
    return np.stack(
        [x_r.flatten(), y_r.flatten(),
         z_r.flatten()], axis=-1) + 0.5


def sample_boundary(bnd, gs):
    # left
    data = sample_box(0, bnd, 0, gs[1], 0, gs[2])
    nor = np.repeat(np.array([[1.0, 0.0, 0.0]], dtype="float32"),
                    bnd * gs[1] * gs[2],
                    axis=0)
    # right
    data = np.append(data,
                     sample_box(gs[0] - bnd, gs[0], 0, gs[1], 0, gs[2]),
                     axis=0)
    nor = np.append(nor,
                    np.repeat(np.array([[-1, 0, 0]], dtype="float32"),
                              bnd * gs[1] * gs[2],
                              axis=0),
                    axis=0)

    # bottom
    data = np.append(data,
                     sample_box(bnd, gs[0] - bnd, 0, bnd, 0, gs[2]),
                     axis=0)
    nor = np.append(nor,
                    np.repeat(np.array([[0, 1, 0]], dtype="float32"),
                              bnd * (gs[0] - 2 * bnd) * gs[2],
                              axis=0),
                    axis=0)
    # top
    data = np.append(data,
                     sample_box(bnd, gs[0] - bnd, gs[1] - bnd, gs[1], 0,
                                gs[2]),
                     axis=0)
    nor = np.append(nor,
                    np.repeat(np.array([[0, -1, 0]], dtype="float32"),
                              bnd * (gs[0] - 2 * bnd) * gs[2],
                              axis=0),
                    axis=0)

    return data, nor


import zstandard as zstd
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_path',
                    type=str,
                    default="datasets/WaterRamps")
parser.add_argument('--out_path',
                    type=str,
                    default="datasets/WaterRamps")
parser.add_argument('--split', type=str, default="train")
parser.add_argument('--block_size', type=int, default=50)
parser.add_argument('--res', type=int, default=65)
parser.add_argument('--dt', type=int, default=0.0025)

args = parser.parse_args()

split = args.split
data_path = args.data_path
out_path = os.path.join(args.out_path, split)
if not os.path.exists(out_path):
    os.makedirs(out_path)
out_path = os.path.join(out_path, "sim_%04d_%02d.msgpack.zst")

block_size = args.block_size
res = args.res
dt = args.dt

fn = get_input_fn(data_path, 1, 'rollout', split)

for di, data in enumerate(fn()):
    pos = data['pos'].numpy()
    ptype = data['type'].numpy()

    pos = np.concatenate([pos, np.zeros_like(pos[..., :1])], axis=-1)
    pos, bnds = pos[:, ptype == 5], pos[:, ptype == 3]
    vel = np.concatenate([pos[1:] - pos[:-1], pos[-1:] - pos[-2:-1]],
                         axis=0) / dt
    bnds = bnds[0]

    if bnds.shape[0] > 0:
        idx_grid = ParticleIdxGrid(bnds * [res, res, 1], [res, res, 1])
        bnds_nor = np.array(
            [idx_grid.get_normal(p * [res, res, 1]) for p in bnds])

    bnds_d, nor_d = sample_boundary(res * 0.1 * 2, [res * 2, res * 2, 1])
    bnds_d = bnds_d / [res * 2, res * 2, 1]

    if bnds.shape[0] > 0:
        bnds = np.concatenate([bnds, bnds_d], 0)
        bnds_nor = np.concatenate([bnds_nor, nor_d], 0)
    else:
        bnds = bnds_d
        bnds_nor = nor_d

    bnds[:, -1] = 0

    for bi in range(pos.shape[0] // block_size):
        mass = np.ones_like(pos[0, ..., :1])
        viscosity = np.zeros_like(pos[0, ..., :1])

        compressor = zstd.ZstdCompressor(level=22)
        with open(out_path % (di, bi), 'wb') as f:
            print('writing sim_%04d to' % di, out_path % (di, bi))
            data = [({
                'box': bnds,
                'box_normals': bnds_nor,
                'frame_id': bi * block_size + i,
                'scene_id': "sim_%04d" % di,
                'pos': pos[bi * block_size + i],
                'vel': vel[bi * block_size + i]
            }) for i in range(block_size)]
            f.write(compressor.compress(msgpack.packb(data,
                                                      use_bin_type=True)))
