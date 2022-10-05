import tensorflow as tf

import open3d.ml.tf as o3dml
import numpy as np
from utils.tools.losses import get_window_func

from .pbf_model import PBFNet

relu = tf.keras.activations.relu


class CConv(PBFNet):
    def __init__(self,
                 name="CConv",
                 layer_channels=[32, 64, 64, 3],
                 window=None,
                 out_activation=None,
                 **kwargs):

        self.layer_channels = layer_channels
        if out_activation == "tanh":
            self.out_activation = tf.keras.activations.tanh
        elif out_activation is None:
            self.out_activation = lambda x: x
        else:
            raise NotImplementedError()

        super().__init__(name=name,
                         channels=layer_channels[0],
                         window=window,
                         **kwargs)

    def setup(self):
        self.convs = []
        self.denses = []
        for i in range(1, len(self.layer_channels)):
            ch = self.layer_channels[i]
            conv = self.get_cconv(name='conv{0}'.format(i),
                                  filters=ch,
                                  activation=None,
                                  window_func=self.window,
                                  ignore_query_points=self.ignore_query_points,
                                  circular=self.circular)
            self.convs.append(conv)
            dense = tf.keras.layers.Dense(units=ch,
                                          name="dense{0}".format(i),
                                          activation=None)
            self.denses.append(dense)

    def forward(self, prev, data, training=True, **kwargs):
        pos, feats = prev[:2]
        pos = pos[0]
        feats = feats[:tf.shape(pos)[0]]

        # compute the extent of the filters (the diameter)
        filter_extent = tf.constant(self.particle_radii[0]) * 2

        ans_convs = [feats]
        for conv, dense in zip(self.convs, self.denses):
            feats = relu(ans_convs[-1])
            ans_conv = conv(feats, pos, pos, filter_extent, None)
            ans_dense = dense(feats)
            if ans_dense.shape[-1] == ans_convs[-1].shape[-1]:
                ans = ans_conv + ans_dense + ans_convs[-1]
            else:
                ans = ans_conv + ans_dense
            ans_convs.append(ans)

        return self.out_activation(ans_convs[-1])
