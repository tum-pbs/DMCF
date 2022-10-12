import tensorflow as tf
import logging
import numpy as np
from tqdm import tqdm
import re
import os
import time
from glob import glob
import time

from datetime import datetime

from os.path import exists, join
from pathlib import Path

from .base_pipeline import BasePipeline

from o3d.utils import make_dir, PIPELINE, LogRecord, get_runid, code2md

from datasets.dataset_reader_physics import get_dataloader, get_rollout, write_results

from utils.tools.losses import density_loss, get_window_func, compute_density, get_window_func, emd_loss
from utils.evaluation_helper import compare_dist, chamfer_distance, distance, merge_dicts

import warnings

warnings.filterwarnings('ignore')

logging.setLogRecordFactory(LogRecord)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class Simulator(BasePipeline):
    """
    Pipeline for trainable simulator. 
    """
    def __init__(self,
                 model,
                 dataset=None,
                 name='Simulator',
                 main_log_dir='./logs/',
                 device='cuda',
                 split='train',
                 **kwargs):
        super().__init__(model=model,
                         dataset=dataset,
                         name=name,
                         main_log_dir=main_log_dir,
                         device=device,
                         split=split,
                         **kwargs)

    @tf.function(experimental_relax_shapes=True)
    def run_inference(self, inputs):
        """
        Run inference on a given data.

        Args:
            data: A raw data.
        Returns:
            Returns the inference results.
        """
        results = []
        for bi in range(len(inputs)):
            pos, vel = self.model(inputs[bi], training=False)
            results.append([pos, vel] + inputs[bi][2:])
        return results

    def run_rollout(self, inputs, timesteps=2):
        """
        Run rollout on a given data.

        Args:
            data: A raw data.
        Returns:
            Returns the inference results.
        """

        inputs = [[
            tf.convert_to_tensor(data['pos'][0]),
            tf.convert_to_tensor(data['vel'][0]),
            tf.convert_to_tensor(data["grav"][0])
            if data["grav"][0] is not None else None, None,
            tf.convert_to_tensor(data["box"][0]),
            tf.convert_to_tensor(data["box_normals"][0])
        ] for data in inputs]
        results = [[] for _ in range(len(inputs))]

        # dummy init
        self.run_inference(inputs[:1])

        timing = []
        for i in range(len(inputs)):
            results[i].append(inputs[i])
        for t in tqdm(range(timesteps - 1), "rollout"):
            start = time.time()
            for i in range(len(inputs)):
                inputs[i] = self.run_inference(inputs[i:i + 1])[0]
            end = time.time()
            timing.append(end - start)
            for i in range(len(inputs)):
                results[i].append(inputs[i])
        log.info("Average runtime: %.05f" % (np.mean(timing) / len(inputs)))

        return results

    def run_test(self, epoch=None):
        """
        Run test with test data split.
        """
        model = self.model
        dataset = self.dataset

        cfg = self.cfg

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log_file_path = join(cfg.logs_dir, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        test_data = get_rollout(dataset.test, **cfg.data_generator,
                                **cfg.data_generator.test)

        if epoch is None:
            epoch = self.load_ckpt(model.cfg.ckpt_path)

        log.info("Started testing")

        results = self.run_rollout(test_data, test_data[0]["pos"].shape[0])

        for i in tqdm(range(len(results)), desc='write out'):
            data = test_data[i]
            pos = np.stack(r[0] for r in results[i])

            out_dir = os.path.join(self.cfg.out_dir, "visual", "%04d" % i)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            output = [(pos, {
                "name": "pred",
                "type": "PARTICLE"
            }), (data['pos'], {
                "name": "gt",
                "type": "PARTICLE"
            }), (data['box'][0], {
                "name": "bnd",
                "type": "PARTICLE"
            })]

            write_results(os.path.join(out_dir, '%04d.hdf5' % epoch),
                          self.model.name, output)

            for f in glob(os.path.join(out_dir, '*.hdf5')):
                if f != os.path.join(out_dir, '%04d.hdf5' % epoch):
                    log.info("Remove %s" %
                             os.path.join(out_dir, '%04d.hdf5' % epoch))
                    os.remove(f)

        if cfg.get('test_compute_metric', False):
            self.run_valid(epoch)

    def run_valid(self, epoch=None):
        model = self.model
        dataset = self.dataset
        cfg = self.cfg

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log_file_path = join(cfg.logs_dir, 'log_valid_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        valid_data = get_rollout(dataset.valid, **cfg.data_generator,
                                 **cfg.data_generator.valid)

        if epoch is None:
            epoch = self.load_ckpt(model.cfg.ckpt_path)

        log.info("Started validation")

        results = self.run_rollout(valid_data, valid_data[0]["pos"].shape[0])

        losses = []
        for i in tqdm(range(len(valid_data)), desc='validation'):
            data = valid_data[i]
            target_pos, target_vel = data["pos"], data["vel"]

            loss_seq = []
            for t in range(1, target_pos.shape[0]):
                # eval for complete sequence
                pos, vel = results[i][t][:2]
                loss = {}
                # loss = model.loss(
                #    [pos, vel], [results[i][t], target_pos[t], target_pos[t - 1], 0])

                # for l, v in cfg.metrics.items():
                #     if v["typ"] == "hist":
                #         feat = v.get("feat", "vel")
                #         if feat == "vel":
                #             loss[l] = np.mean(get_loss(**v)(target_vel[t], vel))
                #     elif v["typ"] == "dense":
                #         loss[l] = np.mean(get_loss(**v)(
                #             target_pos[t], pos,
                #             tf.concat([pos, data["box"][0]], axis=0),
                #             tf.concat([target_pos[t], data["box"][0]], axis=0)))
                #     elif v["typ"] == "chamfer" and v["mode"] > 0:
                #         loss[l] = np.mean(get_loss(**v)(target_pos[t], tf.clip_by_value(pos, -0.5, 0.5)))
                #     else:
                #         loss[l] = np.mean(get_loss(**v)(target_pos[t], pos))

                if t % cfg.data_generator.valid.get("eval_stride", 1) == 0:
                    if data["box"][0].shape[0] > 0:
                        pos = tf.clip_by_value(pos,
                                               np.min(data["box"][0], axis=0),
                                               np.max(data["box"][0], axis=0))
                    loss['mse_val'] = np.mean(distance(target_pos[t], pos))
                    loss['chamfer_val'] = np.mean(
                        chamfer_distance(target_pos[t],
                                         pos).astype(np.float32))

                    if cfg.split != "train":
                        loss['dens_val'] = np.mean(
                            density_loss(target_pos[t],
                                         pos,
                                         tf.concat([pos, data["box"][0]],
                                                   axis=0),
                                         tf.concat(
                                             [target_pos[t], data["box"][0]],
                                             axis=0),
                                         win=get_window_func("poly6")).numpy())
                        loss['max_dens_val'] = density_loss(
                            pos,
                            target_pos[t],
                            tf.concat([pos, data["box"][0]], axis=0),
                            tf.concat([target_pos[t], data["box"][0]], axis=0),
                            radius=model.particle_radii[0],
                            win=get_window_func(model.window_dens),
                            use_max=True).numpy()
                        loss['chamfer_val_2'] = np.mean(
                            chamfer_distance(pos,
                                             target_pos[t]).astype(np.float32))
                        loss['emd'] = np.mean(
                            emd_loss(target_pos[t:t + 1], tf.expand_dims(
                                pos, 0)).numpy().astype(np.float32))

                        loss['vel_diff_val'] = compare_dist(target_vel[t], vel)
                        loss['vel_diff_val_2'] = compare_dist(
                            vel, target_vel[t])

                    # mse for single step only
                    pos_sub = self.model(
                        [target_pos[t - 1], target_vel[t - 1]] +
                        results[i][t][2:])[0]
                    loss['mse_single_val'] = np.mean(
                        distance(target_pos[t], pos_sub))

                    losses.append(loss)
                    loss_seq.append(loss)

            loss_m = merge_dicts(loss_seq, lambda x, y: x + y / len(loss_seq))

            desc = "%d -" % i
            for l, v in loss_m.items():
                desc += " %s: %.05f" % (l, v)

            log.info(desc)

        loss = merge_dicts(losses, lambda x, y: x + y / len(losses))

        sum_loss = 0
        desc = "validation of epoch %d -" % epoch
        for l, v in loss.items():
            desc += " %s: %.05f" % (l, v)
            sum_loss += v
        desc += " > loss: %.05f" % sum_loss
        loss["loss"] = sum_loss

        log.info(desc)

        self.valid_loss = loss

    def run_train(self):
        model = self.model
        dataset = self.dataset

        cfg = self.cfg

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_file_path = join(cfg.logs_dir, 'log_train_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        train_loader = get_dataloader(dataset.train,
                                      batch_size=cfg.batch_size,
                                      pre_frames=cfg.max_warm_up[0],
                                      max_pre_frames=cfg.max_warm_up[-1],
                                      window=cfg.windows[0],
                                      max_window=cfg.windows[-1],
                                      **cfg.data_generator,
                                      **cfg.data_generator.train)
        # wait for dataloader
        time.sleep(10)

        self.optimizer = model.get_optimizer(cfg.optimizer)

        is_resume = model.cfg.get('is_resume', True)
        start_ep = self.load_ckpt(model.cfg.ckpt_path, is_resume=is_resume)

        log.info("Writing summary in {}.".format(self.tensorboard_dir))

        @tf.function(experimental_relax_shapes=True)
        def train(data, time_w, it=0, max_err=None, max_dens_err=None):
            loss = []

            # warmup
            in_pos, in_vel = [], []
            pre = []
            for bi in range(len(data['pos'])):
                acc = data["grav"][bi][0]
                pr_pos = data["pos"][bi][0]
                pr_vel = data["vel"][bi][0]

                p = 0
                prev_err, prev_dens_err = 0.0, 0.0
                for p in range(data['pre'][bi]):
                    inputs = (pr_pos, pr_vel, acc, None, data["box"][bi][0],
                              data["box_normals"][bi][0])
                    pos, vel = model(inputs, training=False)

                    if max_err is not None:
                        err = tf.reduce_max(
                            tf.reduce_sum(tf.abs(pos - data["pos"][bi][p]),
                                          axis=-1))
                        if p > 0 and err > prev_err and err > max_err:
                            break
                        prev_err = err

                    if max_dens_err is not None:
                        err = density_loss(
                            pos,
                            data["pos"][bi][p],
                            tf.concat([pos, data["box"][bi][0]], axis=0),
                            tf.concat([data["pos"][bi][p], data["box"][bi][0]],
                                      axis=0),
                            radius=model.particle_radii[0],
                            win=get_window_func(model.window_dens),
                            use_max=True)

                        if p > 0 and err > prev_dens_err and err > max_dens_err:
                            break
                        prev_dens_err = err

                    pr_pos, pr_vel = pos, vel
                pre.append(p)
                in_pos.append(pr_pos)
                in_vel.append(pr_vel)

            with tf.GradientTape() as tape:
                loss = tf.TensorArray(tf.float32,
                                      size=tf.shape(time_w)[0] *
                                      len(data['pos']),
                                      dynamic_size=True,
                                      clear_after_read=False)
                inputs = []
                for bi in range(len(data['pos'])):

                    def body(pos, vel, pre, t, loss):
                        inputs = [
                            pos, vel, data["grav"][bi][0], None,
                            data["box"][bi][0], data["box_normals"][bi][0]
                        ]
                        target = data["pos"][bi]
                        pos, vel = model(inputs, training=True)
                        l = []
                        l.append(
                            model.loss([pos, vel], [
                                inputs, target[t + pre + 1], target[t + pre],
                                pre
                            ]))
                        for _ in range(1, it):
                            pos, vel = model(inputs, vel, training=True)
                            l.append(
                                model.loss([pos, vel], [
                                    inputs, target[t + pre + 1],
                                    target[t + pre], pre
                                ]))

                        l = merge_dicts(l, lambda x, y: x + y / len(l))
                        loss = loss.write(
                            t + bi * tf.shape(time_w)[0],
                            tf.convert_to_tensor(list(l.values())) * time_w[t])
                        return pos, vel, pre, t + 1, loss

                    loss = tf.while_loop(
                        lambda p, v, pr, t, l: t < tf.shape(time_w)[0], body, [
                            in_pos[bi], in_vel[bi], pre[bi],
                            tf.constant(0), loss
                        ])[-1]
                loss_sum = tf.reduce_sum(loss.stack(), axis=0) / (
                    tf.reduce_sum(time_w) * len(data['pos']))

                w_decay = cfg.get("w_decay", 0)
                if w_decay > 0:
                    loss_sum += w_decay * tf.reduce_sum(
                        [tf.reduce_sum(w**2) for w in model.trainable_weights])

                grads = tape.gradient(loss_sum, model.trainable_weights)

                norm = cfg.get('grad_clip_norm', -1)
                if norm > 0:
                    grads = [tf.clip_by_norm(g, norm) for g in grads]

                self.optimizer.apply_gradients(
                    zip(grads, model.trainable_weights))

            return loss_sum, pre

        window_it, warm_up_it, it_idx = 0, 0, 0
        log.info("Started training")
        for epoch in range(start_ep, cfg.max_epoch + 1):
            log.info(f'=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===')
            process_bar = tqdm(range(cfg.iter), desc='training')
            for i in process_bar:
                step = epoch * cfg.iter + i

                while window_it < min(len(cfg.windows), len(cfg.window_bnds)
                                      ) and step >= cfg.window_bnds[window_it]:
                    window_it += 1
                    train_loader = get_dataloader(
                        dataset.train,
                        batch_size=cfg.batch_size,
                        pre_frames=cfg.max_warm_up[warm_up_it],
                        window=cfg.windows[window_it],
                        **cfg.data_generator,
                        **cfg.data_generator.train)
                    time.sleep(10)

                while warm_up_it < min(
                        len(cfg.max_warm_up), len(cfg.warm_up_bnds)
                ) and step >= cfg.warm_up_bnds[warm_up_it]:
                    warm_up_it += 1
                    train_loader = get_dataloader(
                        dataset.train,
                        batch_size=cfg.batch_size,
                        pre_frames=cfg.max_warm_up[warm_up_it],
                        window=cfg.windows[window_it],
                        **cfg.data_generator,
                        **cfg.data_generator.train)
                    time.sleep(10)

                while it_idx < min(len(cfg.iterations), len(
                        cfg.its_bnds)) and step >= cfg.its_bnds[it_idx]:
                    it_idx += 1

                data_fetch_start = time.time()
                data = next(train_loader)

                time_w = np.ones((np.min([
                    d.shape[0] - 1 - p
                    for d, p in zip(data['pos'], data['pre'])
                ])),
                                 dtype=np.float32)
                if window_it > 0:
                    a = (step - cfg.window_bnds[window_it - 1] +
                         1) / cfg.time_blend
                    if a < 1.0 and len(time_w) >= cfg.windows[window_it]:
                        diff = cfg.windows[window_it] - cfg.windows[window_it -
                                                                    1]
                        time_w[-diff:] = np.clip(a - np.arange(diff) / diff,
                                                 0.0, 1.0)
                time_w = tf.convert_to_tensor(list(time_w))
                tcnt = tf.cast(tf.math.ceil(tf.reduce_sum(time_w)), tf.int32)

                data_fetch_latency = time.time() - data_fetch_start
                self.log_scalar_every_n_minutes(self.writer, step, 5,
                                                'DataLatency',
                                                data_fetch_latency)

                loss_l, pre = train(data, time_w, cfg.iterations[it_idx],
                                    cfg.get('max_err', None),
                                    cfg.get('max_dens_err', None))

                if i == 0 and epoch == start_ep:
                    self.log_param_count()

                loss = {}
                desc = "training -"
                for l, v in zip(model.loss_keys(), loss_l):
                    desc += " %s: %.05f" % (l, v.numpy())
                    loss[l] = v.numpy()
                loss["loss"] = np.sum(loss_l)
                desc += " > loss: %.05f" % loss["loss"]

                loss["timesteps"] = np.minimum(
                    tf.reduce_sum(time_w).numpy(), tcnt.numpy())
                loss["warmup"] = tf.reduce_mean(data['pre'])
                loss["warmup_diff"] = tf.reduce_mean(
                    tf.convert_to_tensor(data['pre']) -
                    tf.convert_to_tensor(pre))

                process_bar.set_description(desc)
                process_bar.refresh()

                self.save_logs(self.writer, step, [loss], "train")

            if epoch % cfg.save_ckpt_freq == 0:
                self.save_ckpt(epoch)

            # --------------------- validation
            self.run_valid(epoch)
            self.save_logs(self.writer, epoch, [self.valid_loss], "valid")

            self.run_test(epoch)
