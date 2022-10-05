import argparse
import copy
import os
import sys
import os.path as osp
from pathlib import Path
import yaml
import time
import pprint
import importlib


def parse_args():
    parser = argparse.ArgumentParser(description='Run a network')
    parser.add_argument('-c', '--cfg_file', help='path to the config file')
    parser.add_argument('-m', '--model', help='network model')
    parser.add_argument('--ckpt_path', help='path to the checkpoint')
    parser.add_argument('--data_path', help='path to the data')
    parser.add_argument('--inflow', help='Inflow timing', default=0, type=int)
    parser.add_argument('--timesteps',
                        help='Amount of timesteps',
                        default=None,
                        type=int)
    parser.add_argument('--device',
                        help='device to run the pipeline',
                        default='gpu')
    parser.add_argument('--output_dir',
                        default="output",
                        help='the dir to save outputs')

    args, unknown = parser.parse_known_args()

    parser_extra = argparse.ArgumentParser(description='Extra arguments')
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser_extra.add_argument(arg)
    args_extra = parser_extra.parse_args(unknown)

    print("regular arguments")
    print(yaml.dump(vars(args)))

    print("extra arguments")
    print(yaml.dump(vars(args_extra)))

    return args, vars(args_extra)


from o3d.utils import convert_device_name, Config
import tensorflow as tf
from datasets import DatasetGroup
import models
from tqdm import tqdm
import re

from datasets.dataset_reader_physics import write_results

import multiprocessing

multiprocessing.set_start_method('spawn', True)

import random
import numpy as np

random.seed(42)
np.random.seed(42)

import zstandard as zstd
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

cmd_line = ' '.join(sys.argv[:])
args, extra_dict = parse_args()


def setup():

    args.device = convert_device_name(args.device)

    device = args.device
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if device == 'cpu':
                tf.config.set_visible_devices([], 'GPU')
            elif device == 'cuda':
                tf.config.set_visible_devices(gpus[0], 'GPU')
            else:
                idx = device.split(':')[1]
                tf.config.set_visible_devices(gpus[int(idx)], 'GPU')
        except RuntimeError as e:
            print(e)

    if args.cfg_file is not None:
        cfg = Config.load_from_file(args.cfg_file)

        Model = getattr(models, cfg.model.name)

        cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = \
                        Config.merge_cfg_file(cfg, args, extra_dict)

        model = Model(**cfg_dict_model)

    else:
        if args.model is None:
            raise ValueError("please specify pipeline, model, and dataset " +
                             "if no cfg_file given")

        Model = getattr(models, args.model)

        cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = \
                        Config.merge_module_cfg_file(args, extra_dict)

        model = Model(**cfg_dict_model)
    return model


@tf.function(experimental_relax_shapes=True)
def run_inference(inputs):
    """
    Run inference on a given data.

    Args:
        data: A raw data.
    Returns:
        Returns the inference results.
    """
    results = []
    for bi in range(len(inputs)):
        pos, vel = model(inputs[bi], training=False)
        results.append([pos, vel] + inputs[bi][2:])
    return results


def run_rollout(data, timesteps=2):
    """
    Run rollout on a given data.

    Args:
        data: A raw data.
    Returns:
        Returns the inference results.
    """
    in_pos = tf.convert_to_tensor(data["pos"])
    in_vel = tf.convert_to_tensor(data["vel"]) + (
        tf.constant([[10.0, 0, -6]]) +
        0 * np.random.normal(scale=(1.0, 0.1, 0.6), size=data["vel"].shape))
    in_acc = tf.zeros_like(in_pos) + tf.constant([[0, model.grav, 0]])

    inputs = [
        in_pos, in_vel, in_acc, None,
        tf.convert_to_tensor(data["box"]),
        tf.convert_to_tensor(data["box_normals"])
    ]
    results = []

    # dummy init
    run_inference([inputs])

    results.append(inputs[0])
    timing = []
    for t in tqdm(range(timesteps - 1), "rollout"):
        start = time.time()
        #print(inputs[0].shape)
        inputs = run_inference([inputs])[0]
        end = time.time()
        timing.append(end - start)
        results.append(inputs[0])

        if args.inflow > t and t % 2 == 1:
            # inflow:
            inputs[0] = tf.concat([inputs[0], in_pos], axis=0)
            inputs[1] = tf.concat([inputs[1], in_vel], axis=0)
            inputs[2] = tf.concat([inputs[2], in_acc], axis=0)

    print("Average runtime: %.05f" % (np.mean(timing)))

    return results


def load_ckpt(ckpt_path, model):
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=100)

    epoch = 0
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        epoch = int(re.findall(r'\d+', manager.latest_checkpoint)[-1])
    else:
        ckpt.restore(
            ckpt_path).expect_partial()  #assert_existing_objects_matched()
        print("Restored from {}".format(ckpt_path))
    return epoch


def load_data(path):
    decompressor = zstd.ZstdDecompressor()
    with open(path, 'rb') as f:
        data = msgpack.unpackb(decompressor.decompress(f.read()), raw=False)
    return data


def main(model):
    data = load_data(args.data_path)
    epoch = load_ckpt(args.ckpt_path, model)

    results = run_rollout(
        data[0],
        len(data) if args.timesteps is None else args.timesteps)

    pos = np.ones((len(results), results[-1].shape[0], 3)) * 1000

    for i in range(len(results)):
        pos[i, :results[i].shape[0]] = results[i]

    print(pos.shape)

    out_dir = os.path.join(args.output_dir, "example", "0000")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    output = [(pos, {
        "name": "pred",
        "type": "PARTICLE"
    }), (data[0]['box'], {
        "name": "bnd",
        "type": "PARTICLE"
    })]

    write_results(os.path.join(out_dir, '%04d.hdf5' % epoch), model.name,
                  output)


if __name__ == '__main__':
    model = setup()
    main(model)
