import numpy as np
import logging
import yaml
import tensorflow as tf
from abc import ABC, abstractmethod
import re
import os
import shutil
import time

from os.path import join, exists, dirname, abspath
from pathlib import Path

from o3d.utils import Config, make_dir, LogRecord, code2md, get_runid

logging.setLogRecordFactory(LogRecord)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class BasePipeline(ABC):
    """
    Base pipeline class
    """
    def __init__(self, model, dataset=None, config=None, **kwargs):
        """
        Initialize

        Args:
            model: network
            dataset: dataset, or None for inference model
            devce: 'gpu' or 'cpu'
            kwargs:
        Returns:
            class: The corresponding class.
        """
        if kwargs['name'] is None:
            raise KeyError("Please give a name to the pipeline")

        self.cfg = Config(kwargs)
        self.name = self.cfg.name
        self.version = self.cfg.version

        self.model = model
        self.dataset = dataset

        make_dir(self.cfg.main_log_dir)
        dataset_name = dataset.name if dataset is not None else ''
        self.cfg.logs_dir = join(
            self.cfg.main_log_dir,
            model.__class__.__name__ + '_' + dataset_name + '_' + self.version)
        if self.cfg.restart and os.path.exists(self.cfg.logs_dir):
            shutil.rmtree(self.cfg.logs_dir)
        make_dir(self.cfg.logs_dir)

        make_dir(self.cfg.output_dir)
        dataset_name = dataset.name if dataset is not None else ''
        self.cfg.out_dir = join(
            self.cfg.output_dir,
            model.__class__.__name__ + '_' + dataset_name + '_' + self.version)
        if self.cfg.restart and os.path.exists(self.cfg.out_dir):
            shutil.rmtree(self.cfg.out_dir)
        make_dir(self.cfg.out_dir)

        if config and join(self.cfg.logs_dir, "config.txt"):
            with open(join(self.cfg.logs_dir, "config.txt"), 'w') as f:
                f.write(config.dump())

        tensorboard_dir = join(
            self.cfg.train_sum_dir,
            model.__class__.__name__ + '_' + dataset_name + '_' + self.version)
        runid = get_runid(tensorboard_dir)
        self.tensorboard_dir = join(self.cfg.train_sum_dir,
                                    runid + '_' + Path(tensorboard_dir).name)

        self.writer = tf.summary.create_file_writer(self.tensorboard_dir)
        self._true_every_n_minutes_last_time = {}

    @abstractmethod
    def run_inference(self, data):
        """
        Run inference on a given data.

        Args:
            data: A raw data.
        Returns:
            Returns the inference results.
        """
        return

    @abstractmethod
    def run_rollout(self, data):
        """
        Run inference on a given data.

        Args:
            data: A raw data.
        Returns:
            Returns the inference results.
        """
        return

    @abstractmethod
    def run_test(self):
        """
        Run testing on test sets.
            
        """
        return

    @abstractmethod
    def run_train(self):
        """
        Run training on train sets
        """
        return

    def log_param_count(self, model=None):
        if model is None:
            model = self.model
        trainable_count = np.sum(
            tf.keras.backend.count_params(w) for w in model.trainable_weights)
        non_trainable_count = np.sum(
            tf.keras.backend.count_params(w)
            for w in model.non_trainable_weights)
        log.info("###################################")
        log.info("Parameter count '{}':".format(model.name))
        log.info(" Total params: {:,}".format(trainable_count +
                                              non_trainable_count))
        log.info(" Trainable params: {:,}".format(trainable_count))
        log.info(" Non-trainable params: {:,}".format(non_trainable_count))
        log.info("-----------------------------------")

    def save_logs(self, writer, epoch, data, prefix=""):
        with writer.as_default():
            for d in data:
                for key, val in d.items():
                    tf.summary.scalar(os.path.join(prefix, key), val, epoch)

            if self.optimizer is not None:
                if isinstance(self.optimizer, list):
                    for i in range(len(self.optimizer)):
                        tf.summary.scalar('train/learning_rate_%d' % i,
                                          self.optimizer[i]._decayed_lr(
                                              tf.float32),
                                          step=epoch)
                else:
                    tf.summary.scalar('train/learning_rate',
                                      self.optimizer._decayed_lr(tf.float32),
                                      step=epoch)

    def load_ckpt(self, ckpt_path=None, is_resume=True):
        train_ckpt_dir = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(train_ckpt_dir)

        if hasattr(self, 'optimizer'):
            self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                            optimizer=self.optimizer,
                                            model=self.model)
        else:
            self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                            model=self.model)

        self.manager = tf.train.CheckpointManager(self.ckpt,
                                                  train_ckpt_dir,
                                                  max_to_keep=100)

        epoch = 0
        if ckpt_path is not None:
            self.ckpt.restore(
                ckpt_path).expect_partial()  #assert_existing_objects_matched()
            log.info("Restored from {}".format(ckpt_path))
        else:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()

            if self.manager.latest_checkpoint:
                log.info("Restored from {}".format(
                    self.manager.latest_checkpoint))
                epoch = int(
                    re.findall(r'\d+', self.manager.latest_checkpoint)[-1]) - 1
                epoch = epoch * self.cfg.save_ckpt_freq + 1
            else:
                log.info("Initializing from scratch.")
        return epoch

    def save_ckpt(self, epoch):
        save_path = self.manager.save()
        log.info("Saved checkpoint at: {}".format(save_path))

    def _true_every_n_minutes(self, n, name):
        now = time.time()

        key = (n, name)
        if not key in self._true_every_n_minutes_last_time:
            self._true_every_n_minutes_last_time[key] = now
            return True
        else:
            last = self._true_every_n_minutes_last_time[key]
            if now - last > 60 * n:
                self._true_every_n_minutes_last_time[key] = now
                return True

        return False

    def log_scalar_every_n_minutes(self, writer, step, n, name, value):
        """Convenience function for calling tf.summary.scalar in regular time intervals."""
        if self._true_every_n_minutes(n, name):
            with writer.as_default():
                tf.summary.scalar(name, value, step=step)
