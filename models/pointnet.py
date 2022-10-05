import tensorflow as tf

import open3d.ml.tf as o3dml
import numpy as np
from utils.tools.losses import get_window_func
from utils.tools.losses import get_dilated_pos

from .pbf_model import PBFNet

relu = tf.keras.activations.relu


class PointNet(PBFNet):
    def __init__(self,
                 name="CConv",
                 layer_channels=[32, 64, 64, 3],
                 out_activation=None,
                 **kwargs):

        self.layer_channels = layer_channels
        if out_activation == "tanh":
            self.out_activation = tf.keras.activations.tanh
        elif out_activation is None:
            self.out_activation = lambda x: x
        else:
            raise NotImplementedError()

        super().__init__(name=name, channels=layer_channels[0], **kwargs)

    def setup(self):
        self.denses = []
        for i in range(len(self.layer_channels)):
            dense = tf.keras.layers.Dense(units=self.layer_channels[i],
                                          name="dense{0}".format(i),
                                          activation=None)
            self.denses.append(dense)

    def preprocess(self,
                   data,
                   training=True,
                   vel_corr=None,
                   tape=None,
                   **kwargs):
        #
        # advection step
        #
        _pos, _vel, acc, feats, box, bfeats = data

        if vel_corr is not None:
            vel = tf.stop_gradient(vel_corr)
            pos = _pos + vel * self.timestep
        else:
            pos, vel = self.integrate_pos_vel(_pos, _vel, acc)

        #
        # preprocess features
        #
        # compute the extent of the filters (the diameter)
        filter_extent = tf.constant(self.particle_radii) * 2

        fluid_feats = [tf.ones_like(pos[:, :1])]
        if self.use_vel:
            fluid_feats.append(vel)
        if self.use_acc:
            fluid_feats.append(acc)
        if self.use_feats:
            fluid_feats.append(feats)
        box_feats = [tf.ones_like(box[:, :1])]
        if self.use_box_feats:
            box_feats.append(bfeats)

        all_pos = tf.concat([pos, box], axis=0)
        self.all_pos = all_pos
        if self.dens_feats or self.dens_norm or self.pres_feats:
            dens = compute_density(all_pos,
                                   all_pos,
                                   self.dens_radius[0],
                                   win=get_window_func(self.window_dens))
            if self.dens_feats:
                fluid_feats.append(tf.expand_dims(dens[:tf.shape(pos)[0]], -1))
                box_feats.append(tf.expand_dims(dens[tf.shape(pos)[0]:], -1))
            if self.pres_feats:
                pres = compute_pressure(all_pos,
                                        all_pos,
                                        dens,
                                        self.rest_dens,
                                        win=get_window_func(self.window_dens),
                                        stiffness=self.stiffness)
                fluid_feats.append(tf.expand_dims(pres[:tf.shape(pos)[0]], -1))
                box_feats.append(tf.expand_dims(pres[tf.shape(pos)[0]:], -1))

        fluid_feats = tf.concat(fluid_feats, axis=-1)
        box_feats = tf.concat(box_feats, axis=-1)

        self.inp_feats = fluid_feats
        self.inp_bfeats = box_feats
        if tape is not None:
            tape.watch(self.inp_feats)
            tape.watch(self.inp_bfeats)

        dilated_pos, _, idx = get_dilated_pos(
            all_pos if self.use_bnds else pos,
            self.strides,
            voxel_size=self.voxel_size,
            centralize=self.centralize,
            pad=self.sample_pad,
            hyst=self.sample_hyst)

        if self.dens_norm:
            dens = [
                tf.expand_dims(
                    dens if self.use_bnds else dens[:tf.shape(pos)[0]],
                    axis=-1)
            ]
            for scale in range(1, len(self.dens_radius)):
                d = self.sampling(dens[-1], dilated_pos[scale - 1],
                                  dilated_pos[scale], self.dens_radius[scale],
                                  None)
                d = tf.maximum(d, 1e-2)
                dens.append(d)
        else:
            dens = None

        self.dilated_pos = dilated_pos
        return [dilated_pos, fluid_feats, idx, dens]

    def forward(self, prev, data, training=True, **kwargs):
        pos, feats = prev[:2]
        pos = pos[0]

        fixed_radius_search = o3dml.layers.FixedRadiusSearch()
        self.neighbors_index, self.neighbors_row_splits, dist = fixed_radius_search(
            pos, pos, tf.constant(self.particle_radii[0]))

        ans = [feats]
        for dense in self.denses:
            feats = relu(ans[-1])
            ans_dense = dense(feats)
            ans_dense = tf.RaggedTensor.from_row_splits(
                values=tf.gather(ans_dense, self.neighbors_index),
                row_splits=self.neighbors_row_splits)
            ans_dense = tf.reduce_sum(ans_dense, axis=1)
            if ans_dense.shape[-1] == ans[-1].shape[-1]:
                ans_dense = ans_dense + ans[-1]
            ans.append(ans_dense)

        return self.out_activation(ans[-1])

    def postprocess(self, prev, data, training=True, vel_corr=None, **kwargs):
        #
        # postprocess output of network
        #
        pos, vel, acc = data[:3]

        pcnt = tf.shape(pos)[0]

        # compute the number of fluid neighbors.
        # this info is used in the loss function during training.
        self.num_fluid_neighbors = o3dml.ops.reduce_subarrays_sum(
            tf.ones_like(self.neighbors_index, dtype=tf.float32),
            self.neighbors_row_splits)[:pcnt]

        # scale to better match the scale of the output distribution
        out = prev
        if self.equivar:
            scale = self.scale_dens(out)
            rot = None
            #rot = self.rot_dens(out)
            #rot = rot / tf.linalg.norm(rot, axis=-1, keepdims=True)
            out = compute_transformed_dx(self.all_pos,
                                         scale,
                                         rot,
                                         radius=self.particle_radii[0])
        #else:
        #    if out.shape[-1] == 1:
        #        out = tf.repeat(out, 3, axis=-1)
        #    elif out.shape[-1] == 2:
        #        out = tf.concat([out, out[:, :1]], axis=-1)

        self.pos_correction = self.out_scale * out[:pcnt]
        self.obs = self.out_scale * out[pcnt:]

        #
        # correct position and velocity
        #
        if vel_corr is not None:
            vel2 = tf.stop_gradient(vel_corr)
            pos2 = pos + vel2 * self.timestep
        else:
            pos2, vel2 = self.integrate_pos_vel(pos, vel, acc)

        pos2_corrected, vel2_corrected = self.compute_new_pos_vel(
            pos, vel, pos2, vel2, self.pos_correction)

        return [pos2_corrected, vel2_corrected]
