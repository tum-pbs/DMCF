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
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('-c', '--cfg_file', help='path to the config file')
    parser.add_argument('-m', '--model', help='network model')
    parser.add_argument('-p',
                        '--pipeline',
                        help='pipeline',
                        default='DefaultGenerator')
    parser.add_argument('-d', '--dataset', help='dataset')
    parser.add_argument('--cfg_model', help='path to the model\'s config file')
    parser.add_argument('--cfg_pipeline',
                        help='path to the pipeline\'s config file')
    parser.add_argument('--cfg_dataset',
                        help='path to the dataset\'s config file')
    parser.add_argument('--dataset_path', help='path to the dataset')
    parser.add_argument('--ckpt_path', help='path to the checkpoint')
    parser.add_argument('--device',
                        help='device to run the pipeline',
                        default='gpu')
    parser.add_argument('--split', help='train or test', default='train')
    parser.add_argument('--regen',
                        help='Regenerates data, overwrites cache',
                        default=False,
                        action='store_true')
    parser.add_argument(
        '--restart',
        help='Restart training, resume if false (overwrites checkpoints!)',
        default=False,
        action='store_true')
    parser.add_argument('--main_log_dir',
                        help='the dir to save logs and models')
    parser.add_argument('--output_dir', help='the dir to save outputs')

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
import pipelines
import models

import multiprocessing

multiprocessing.set_start_method('spawn', True)

import random
import numpy as np

random.seed(42)
np.random.seed(42)


def main():
    cmd_line = ' '.join(sys.argv[:])
    args, extra_dict = parse_args()

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

        Pipeline = getattr(pipelines, cfg.pipeline.name)
        Model = getattr(models, cfg.model.name)

        cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = \
                        Config.merge_cfg_file(cfg, args, extra_dict)

        dataset = DatasetGroup(**cfg_dict_dataset,
                               split=args.split,
                               regen=args.regen)
        model = Model(**cfg_dict_model)
        pipeline = Pipeline(model,
                            dataset,
                            **cfg_dict_pipeline,
                            config=cfg,
                            restart=args.restart)

    else:
        if (args.pipeline and args.model and args.dataset) is None:
            raise ValueError("please specify pipeline, model, and dataset " +
                             "if no cfg_file given")

        Pipeline = getattr(pipelines, args.pipeline)
        Model = getattr(models, args.model)


        cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = \
                        Config.merge_module_cfg_file(args, extra_dict)

        dataset = DatasetGroup(**cfg_dict_dataset,
                               split=args.split,
                               regen=args.regen)
        model = Model(**cfg_dict_model)
        pipeline = Pipeline(model,
                            dataset,
                            **cfg_dict_pipeline,
                            restart=args.restart)

    pipeline.cfg_tb = {
        'cmd_line': cmd_line,
        'dataset': pprint.pformat(cfg_dict_dataset, indent=2),
        'model': pprint.pformat(cfg_dict_model, indent=2),
        'pipeline': pprint.pformat(cfg_dict_pipeline, indent=2)
    }

    if args.split == 'test':
        pipeline.run_test()
    elif args.split == 'valid':
        pipeline.run_valid()
    else:
        pipeline.run_train()


if __name__ == '__main__':
    main()
