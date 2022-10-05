import tensorflow as tf

import open3d.ml.tf as o3dml
import numpy as np
from utils.tools.losses import get_window_func

from .hrnet import HRNet

relu = tf.keras.activations.relu


class SymNet(HRNet):
    def __init__(self,
                 name="SymNet",
                 layer_channels=[[[16]], [[32]], [[32]], [[3]]],
                 sym_kernel_size=[6, 6, 6],
                 sym_axis=2,
                 window_sym=None,
                 out_activation=None,
                 **kwargs):

        self.sym_kernel_size = sym_kernel_size
        self.sym_axis = sym_axis
        self.window_sym = window_sym
        self.sym_channels = layer_channels[-1][-1]

        if out_activation == "tanh":
            self.act = tf.keras.activations.tanh
        elif out_activation is None:
            self.act = lambda x: x
        else:
            raise NotImplementedError()

        super().__init__(name=name,
                         layer_channels=layer_channels[:-1],
                         out_activation=None,
                         **kwargs)

    def setup(self):
        super().setup()
        self.sym_convs = []
        for i, ch in enumerate(self.sym_channels):
            conv = self.get_cconv(name='sym_conv{0}'.format(i),
                                  filters=ch,
                                  activation=None,
                                  use_bias=False,
                                  symmetric=True,
                                  kernel_size=self.sym_kernel_size,
                                  ignore_query_points=True,
                                  window_func=self.window_sym,
                                  sym_axis=self.sym_axis,
                                  circular=self.circular)
            self.sym_convs.append(conv)

    def forward(self, prev, data, training=True, **kwargs):
        pos, feats, idx, dens = prev

        ans = super().forward(prev, data, training, **kwargs)

        if not self.use_bnds:
            ans = tf.concat([ans, feats[tf.shape(pos[0])[0]:]], axis=0)

        ext = tf.constant(self.particle_radii[0]) * 2
        for conv in self.sym_convs:
            ans = tf.keras.activations.relu(ans)
            ans = conv(ans * self.part_scale, self.all_pos, self.all_pos, ext,
                       None)

        return self.act(ans)
