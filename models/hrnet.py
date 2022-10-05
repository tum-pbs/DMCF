import tensorflow as tf

import open3d.ml.tf as o3dml
import numpy as np
from utils.tools.losses import get_window_func

from .pbf_model import PBFNet

relu = tf.keras.activations.relu


class HRNet(PBFNet):
    def __init__(self,
                 name="HRNet",
                 layer_channels=[[16], [32], [32], [3]],
                 window=None,
                 window_dens=None,
                 circular=False,
                 add_merge=False,
                 out_activation=None,
                 **kwargs):

        self.layer_channels = layer_channels
        self.add_merge = add_merge
        if out_activation == "tanh":
            self.out_activation = tf.keras.activations.tanh
        elif out_activation is None:
            self.out_activation = lambda x: x
        else:
            raise NotImplementedError()

        super().__init__(name=name,
                         channels=layer_channels[0][0][0],
                         window=window,
                         window_dens=window_dens,
                         circular=circular,
                         **kwargs)

    def setup(self):
        self.convs = []
        self.denses = []
        for i in range(1, len(self.layer_channels)):
            self.denses.append([])
            self.convs.append([])
            for j in range(len(self.layer_channels[i])):
                self.convs[-1].append([])
                self.denses[-1].append([])
                for k in range(len(self.layer_channels[i][j])):
                    ch = self.layer_channels[i][j][k]
                    self.convs[-1][-1].append([])
                    self.denses[-1][-1].append([])
                    for l in range(
                            len(self.layer_channels[i - 1]) if k == 0 else 1):
                        conv = self.get_cconv(
                            name='conv{0}{1}{2}_{3}'.format(i, j, k, l),
                            filters=ch,
                            activation=None,
                            window_func=self.window,
                            ignore_query_points=self.ignore_query_points
                            and (j == l or k > 0),
                            circular=self.circular)
                        self.convs[-1][-1][-1].append(conv)
                        dense = tf.keras.layers.Dense(
                            units=ch,
                            name="dense{0}{1}{2}_{3}".format(i, j, k, l),
                            activation=None)
                        self.denses[-1][-1][-1].append(dense)

    def forward(self, prev, data, training=True, **kwargs):
        pos, feats, idx, dens = prev

        if not self.use_bnds:
            feats = feats[:tf.shape(pos[0])[0]]

        # compute the extent of the filters (the diameter)
        filter_extent = tf.constant(self.particle_radii) * 2

        ans_convs = [[feats]]
        for layer in range(len(self.convs)):
            ans = []
            for scale in range(len(self.convs[layer])):
                importance = self.part_scale if scale == 0 else 1.0
                inp = []
                for inp_scale in range(len(ans_convs[-1])):
                    feats = relu(ans_convs[-1][inp_scale])
                    ext = filter_extent[max(inp_scale, scale)]
                    if self.dens_norm and inp_scale < len(dens):
                        feats = tf.concat([feats, feats / dens[inp_scale]**2],
                                          axis=-1)
                    ans_conv = self.convs[layer][scale][0][inp_scale](
                        feats * importance, pos[inp_scale], pos[scale], ext,
                        None)
                    if layer < len(self.denses):
                        if scale == inp_scale:
                            ans_conv += self.denses[layer][scale][0][
                                inp_scale](feats)
                            if ans_conv.shape[-1] == ans_convs[-1][
                                    scale].shape[-1]:
                                ans_conv += ans_convs[-1][scale]
                        elif self.voxel_size is None:
                            if scale > inp_scale:
                                for i in range(inp_scale, scale):
                                    feats = tf.gather(feats, idx[i + 1][0])
                                ans_conv += self.denses[layer][scale][0][
                                    inp_scale](feats)
                            else:
                                ind = idx[scale + 1][0]
                                for i in range(scale + 1, inp_scale):
                                    ind = tf.gather(ind, idx[i + 1][0])
                                ans_conv = tf.tensor_scatter_nd_add(
                                    ans_conv, tf.expand_dims(ind, -1),
                                    self.denses[layer][scale][0][inp_scale](
                                        feats))
                    inp.append(ans_conv)
                if self.add_merge:
                    ans.append(tf.math.add_n(inp))
                else:
                    ans.append(tf.concat(inp, axis=-1))

                for i in range(1, len(self.convs[layer][scale])):
                    ans_conv = self.convs[layer][scale][i][0](
                        ans[-1] * importance, pos[scale], pos[scale], ext,
                        None)
                    ans_dens = self.denses[layer][scale][i][0](ans[-1])
                    ans_conv += ans_dens
                    if len(ans_convs[-1]) > scale and ans_conv.shape[
                            -1] == ans_convs[-1][scale].shape[-1]:
                        ans_conv += ans_convs[-1][scale]
                    ans[-1] = ans_conv

            ans_convs.append(ans)

        return self.out_activation(ans_convs[-1][0])
