import tensorflow as tf

import open3d.ml.tf as o3dml
import numpy as np
from abc import abstractmethod
from utils.tools.losses import get_loss, compute_density, compute_pressure, get_window_func, get_dilated_pos, compute_transformed_dx
from utils.convolutions import ContinuousConv, PointSampling

from .base_model import BaseModel


@tf.function
def align_vector(v0, v1):
    v0_norm = v0 / (tf.norm(v0) + 1e-9)
    v1_norm = v1 / (tf.norm(v1) + 1e-9)

    v = tf.linalg.cross(v0_norm, v1_norm)
    c = tf.tensordot(v0_norm, v1_norm, 1)
    s = tf.norm(v)

    if s < 1e-6:
        return tf.eye(3) * (-1.0 if c < 0 else 1.0)

    vx = tf.convert_to_tensor([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]],
                               [-v[1], v[0], 0.0]])

    r = tf.eye(3) + vx + tf.tensordot(vx, vx, 1) / (1 + c)
    return r


class PBFNet(BaseModel):
    def __init__(self,
                 name="PBFNet",
                 kernel_size=[4, 4, 4],
                 channels=16,
                 strides=[1],
                 particle_radii=[0.05],
                 coordinate_mapping='ball_to_cube_volume_preserving',
                 interpolation='linear',
                 window=None,
                 window_dens=None,
                 ignore_query_points=False,
                 grav=-9.81,
                 transformation={},
                 loss={
                     "weighted_mse": {
                         "typ": "weighted_mse",
                         "fac": 1.0,
                         "gamma": 0.25,
                         "neighbor_scale": 0.025
                     }
                 },
                 timestep=0.01,
                 dens_radius=None,
                 circular=False,
                 dens_feats=False,
                 pres_feats=False,
                 equivar=False,
                 use_vel=True,
                 use_acc=True,
                 use_feats=False,
                 use_box_feats=True,
                 use_pre_adv=False,
                 use_bnds=True,
                 dens_norm=False,
                 rest_dens=3.5,
                 stiffness=20.0,
                 voxel_size=None,
                 centralize=False,
                 out_scale=[0.01, 0.01, 0.01],
                 sample_pad=0,
                 sample_hyst=0.1,
                 part_scale=1.0,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        if dens_radius is None:
            dens_radius = particle_radii

        # NN setup
        self.kernel_size = kernel_size
        self.channel = channels
        self.strides = strides
        self.particle_radii = particle_radii
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.window = window
        self.ignore_query_points = ignore_query_points
        self.voxel_size = tf.constant(
            voxel_size) if voxel_size is not None else None
        self.centralize = centralize
        self.circular = circular
        self.equivar = equivar

        self.transformation = transformation

        self.sample_pad = sample_pad
        self.sample_hyst = sample_hyst

        # feats setup
        self.dens_radius = dens_radius
        self.dens_feats = dens_feats
        self.pres_feats = pres_feats
        self.out_scale = tf.constant(out_scale)
        self.window_dens = window_dens
        self.dens_norm = dens_norm
        self.rest_dens = rest_dens
        self.stiffness = stiffness

        self.use_vel = use_vel
        self.use_acc = use_acc
        self.use_feats = use_feats
        self.use_box_feats = use_box_feats
        self.use_pre_adv = use_pre_adv

        self.use_bnds = use_bnds

        # physics setup
        self.timestep = timestep
        self.grav = grav

        self.part_scale = part_scale

        # loss setup
        self.num_fluid_neighbors = 1
        self.loss_fn = {}
        for l, v in loss.items():
            if v["typ"] == "dense":
                if not "radius" in v:
                    v["radius"] = dens_radius[0]
            self.loss_fn[l] = get_loss(**v)

        self._all_convs = []

        self.fluid_convs = self.get_cconv(name='fluid_obs',
                                          filters=channels,
                                          activation=None,
                                          window_func=self.window,
                                          circular=circular)

        self.fluid_dense = tf.keras.layers.Dense(units=channels,
                                                 name="fluid_dense",
                                                 activation=None)

        self.obs_convs = self.get_cconv(name='obs_conv',
                                        filters=channels,
                                        activation=None,
                                        window_func=self.window,
                                        circular=circular)

        self.obs_dense = tf.keras.layers.Dense(units=channels,
                                               name="obs_dense",
                                               activation=None)

        if self.use_pre_adv:
            self.adv_convs = [
                self.get_cconv(name='adv_conv0',
                               filters=channels,
                               activation=None,
                               window_func=self.window,
                               circular=circular),
                self.get_cconv(name='adv_conv1',
                               filters=channels,
                               activation=None,
                               window_func=self.window,
                               circular=circular)
            ]

            self.adv_dense = [
                tf.keras.layers.Dense(units=channels,
                                      name="adv_dense0",
                                      activation=None),
                tf.keras.layers.Dense(units=channels,
                                      name="adv_dense1",
                                      activation=None)
            ]

        if self.dens_norm:
            self.sampling = PointSampling(name='sampling',
                                          window_function=get_window_func(
                                              self.window_dens),
                                          normalize=True)

        if self.equivar:
            self.scale_dens = tf.keras.layers.Dense(units=1,
                                                    name="scale",
                                                    activation=None)
            self.rot_dens = tf.keras.layers.Dense(units=4,
                                                  name="rot",
                                                  activation=None)

        self.setup()

    @abstractmethod
    def setup(self):
        return

    def get_cconv(self,
                  name,
                  kernel_size=None,
                  activation=None,
                  ignore_query_points=None,
                  window_func=None,
                  normalize=False,
                  **kwargs):

        if kernel_size is None:
            kernel_size = self.kernel_size
        if ignore_query_points is None:
            ignore_query_points = self.ignore_query_points
        conv = ContinuousConv(
            name=name,
            kernel_size=kernel_size,
            activation=activation,
            align_corners=True,
            interpolation=self.interpolation,
            coordinate_mapping=self.coordinate_mapping,
            normalize=normalize,
            window_function=get_window_func(window_func),
            radius_search_ignore_query_points=ignore_query_points,
            use_dense_layer_for_center=False,
            **kwargs)

        self._all_convs.append((name, conv))
        return conv

    def _integrate_pos_vel(self, pos1, vel1, acc1=None):
        """Apply gravity and integrate position and velocity"""
        dt = self.timestep
        vel2 = vel1 + dt * (acc1 if acc1 is not None else tf.constant(
            [0, self.grav, 0]))
        pos2 = pos1 + dt * vel1 + (vel1 + vel2) / 2
        return pos2, vel2

    def integrate_pos_vel(self, pos1, vel1, acc1=None):
        """Apply gravity and integrate position and velocity"""
        dt = self.timestep
        vel2 = vel1 + dt * (acc1 if acc1 is not None else tf.constant(
            [0, self.grav, 0]))
        pos2 = pos1 + dt * vel2
        return pos2, vel2

    def compute_new_pos_vel(self, pos1, vel1, pos2, vel2, pos_correction):
        """Apply the correction
        pos1,vel1 are the positions and velocities from the previous timestep
        pos2,vel2 are the positions after applying gravity and the integration step
        """
        dt = self.timestep
        pos = pos2 + pos_correction
        vel = (pos - pos1) / dt
        return pos, vel

    def transform(self, data, training=True, **kwargs):
        pos, vel, acc, feats, box, bfeats = data

        if "translate" in self.transformation:
            translate = tf.constant(self.transformation["translate"],
                                    tf.float32)
            pos += translate
            box += translate

        if "scale" in self.transformation:
            scale = tf.constant(self.transformation["scale"], tf.float32)
            pos *= scale
            box *= scale
            vel *= scale
            if acc is not None:
                acc *= scale

        if "grav_eqvar" in self.transformation:
            grav_eqvar = tf.constant(self.transformation["grav_eqvar"],
                                     tf.float32)
            # WARNING: assuming same gravity for all particles for one sequence
            self.R = align_vector(grav_eqvar, acc[0])
            pos = tf.linalg.matmul(pos, self.R)
            vel = tf.linalg.matmul(vel, self.R)
            acc = tf.linalg.matmul(acc, self.R)
            box = tf.linalg.matmul(box, self.R)
            bfeats = tf.linalg.matmul(bfeats, self.R)

        return [pos, vel, acc, feats, box, bfeats]

    def inv_transform(self, prev, data, **kwargs):
        pos, vel = prev

        if "grav_eqvar" in self.transformation:
            # WARNING: assuming same gravity for all particles for one sequence
            R = tf.transpose(self.R)
            pos = tf.linalg.matmul(pos, R)
            vel = tf.linalg.matmul(vel, R)

        if "scale" in self.transformation:
            scale = tf.constant(self.transformation["scale"], tf.float32)
            pos /= tf.maximum(scale, 1e-5)
            vel /= tf.maximum(scale, 1e-5)

        if "translate" in self.transformation:
            translate = tf.constant(self.transformation["translate"],
                                    tf.float32)
            pos -= translate

        return pos, vel

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
            # TODO:
            #vel2 *= 0
            #if acc is not None:
            #   acc *= 0

        #
        # preprocess features
        #
        # compute the extent of the filters (the diameter)
        filter_extent = tf.constant(self.particle_radii) * 2

        fltr = tf.reduce_all([
            box >= tf.reduce_min(pos, axis=0) - filter_extent[-1],
            box <= tf.reduce_max(pos, axis=0) + filter_extent[-1]
        ],
                             axis=(0, 2))
        box = box[fltr]
        bfeats = bfeats[fltr]

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

        ans_conv = self.fluid_convs(fluid_feats * self.part_scale, pos,
                                    all_pos, filter_extent[0], None)
        ans_dense = self.fluid_dense(fluid_feats)

        ans_obs = self.obs_convs(box_feats * self.part_scale, box, all_pos,
                                 filter_extent[0], None)
        ans_dense_obs = self.obs_dense(box_feats)

        ans_dense = tf.concat([ans_dense, ans_dense_obs], axis=0)

        if self.use_pre_adv:
            _all_pos = tf.concat([_pos, box], axis=0)
            pre_adv_feats = tf.ones_like(_pos[:, :1])
            if self.use_vel:
                pre_adv_feats = tf.concat([pre_adv_feats, _vel], axis=-1)

            ans_adv = self.adv_convs[0](pre_adv_feats * self.part_scale, _pos,
                                        all_pos, filter_extent[0], None)
            ans_dens_adv = self.adv_dense[0](pre_adv_feats)
            ans_dens_adv = tf.concat([ans_dens_adv, ans_dense_obs], axis=0)
            fluid_feats = tf.concat(
                [ans_conv, ans_obs, ans_adv, ans_dense, ans_dens_adv], axis=-1)

            # ans_adv = self.adv_convs[0](pre_adv_feats, _pos, _pos, filter_extent[0], None)
            # ans_dens_adv = self.adv_dense[0](pre_adv_feats)
            # ans_adv += ans_dens_adv

            # ans_dens_adv = self.adv_dense[1](ans_adv)
            # ans_adv = self.adv_convs[1](ans_adv, _pos, all_pos, filter_extent[0], None)

            # ans_dens_adv = tf.concat([ans_dens_adv, ans_dense_obs], axis=0)
            # fluid_feats = tf.concat([ans_conv, ans_obs, ans_adv, ans_dense, ans_dens_adv], axis=-1)
        else:
            fluid_feats = tf.concat([ans_conv, ans_obs, ans_dense], axis=-1)

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

    def postprocess(self, prev, data, training=True, vel_corr=None, **kwargs):
        #
        # postprocess output of network
        #
        pos, vel, acc = data[:3]

        pcnt = tf.shape(pos)[0]

        # compute the number of fluid neighbors.
        # this info is used in the loss function during training.
        self.num_fluid_neighbors = o3dml.ops.reduce_subarrays_sum(
            tf.ones_like(self.fluid_convs.nns.neighbors_index,
                         dtype=tf.float32),
            self.fluid_convs.nns.neighbors_row_splits)[:pcnt]

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

        if out.shape[-1] == 1:
            out = tf.repeat(out, 3, axis=-1)
        elif out.shape[-1] == 2:
            out = tf.concat([out, out[:, :1]], axis=-1)

        #
        # scale to better match the scale of the output distribution
        #
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

    def loss_keys(self):
        return self.loss_fn.keys()

    def loss(self, results, data):
        loss = {}

        pred = results[0]
        target = data[1]

        for n, l in self.loss_fn.items():
            loss[n] = l(target,
                        pred,
                        num_fluid_neighbors=self.num_fluid_neighbors,
                        input=data[0],
                        target_prev=data[2],
                        pre_steps=data[3],
                        pos_correction=self.pos_correction)

        return loss

    def get_optimizer(self, cfg):
        learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            cfg['lr_boundaries'], cfg['lr_values'])

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate_fn,
                                       epsilon=1e-6)
        return optimizer
