# Deep Momentum-Conserving Fluids (DMCF)

![TensorFlow badge](https://img.shields.io/badge/TensorFlow-supported-brightgreen?style=flat&logo=tensorflow)

This repository contains the code for our NeurIPS paper *Guaranteed Conservation of Momentum for Learning Particle-based Fluid Dynamics*. Our algorithm makes it possible to learn highly accurate, efficient and momentum conserving fluid simulations based on particles.
With the code published here, evaluations from the paper can be reconstructed, and new models can be trained.

<p align="center"> <img src="canyon.gif" alt="canyon video"> </p>

Please cite [our paper](https://openreview.net/pdf?id=6niwHlzh10U) if you find this code useful:
```
@inproceedings{Prantl2022Conserving,
        title     = {Guaranteed Conservation of Momentum for Learning Particle-based Fluid Dynamics},
        author    = {Lukas Prantl and Benjamin Ummenhofer and Vladlen Koltun and Nils Thuerey},
        booktitle = {Conference on Neural Information Processing Systems},
        year      = {2022},
}
```

## Dependencies and Setup

Used environment: python3.7 with CUDA 11.3 and CUDNN 8.0.
- Install libcap-dev: ```sudo apt install libcap-dev```
- Install requirements: ```pip install -r requirements.txt```
- Tensorpack DataFlow ```pip install --upgrade git+https://github.com/tensorpack/dataflow.git```

Optional: 
- Build FPS/EMD module ```cd utils; make; cd ..```
- Install skia for visualization: ```python -m pip install skia-python```

## Datasets

- *WBCSPH2D*: https://mega.nz/folder/m9QF2IJI#lMjsXcmE8_nN7JgLwp5vAw
- *Liquid3D* ([source](https://github.com/isl-org/DeepLagrangianFluids)): https://drive.google.com/file/d/1_-aAd_GHX8StyKWZLpvSWeGQ3vyytf7L
- *WaterRamps* ([source](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate)): ```bash download_waterramps.sh PATH/TO/OUTPUT_DIR```

## Pretrained Models:

The pretrained models are in the ```checkpoints``` subfolder.
Run a pretrained mode by setting the path to the checkpoint with the ```ckpt_path``` argument.
For example:
```bash
python run_pipeline.py --cfg_file configs/symnet.yml \
                       --dataset_path PATH/TO/DATASET \
                       --ckpt_path checkpoints/symnet/ckpt \
                       --split test
```

## Training

Simple 1D test run (data will be generated):
```bash
python run_pipeline.py --cfg_file configs/column/symnet.yml \
                       --split train
```

Run with 2D pipeline:
```bash
python run_pipeline.py --cfg_file configs/symnet.yml \
                       --dataset_path PATH/TO/DATASET \
                       --split train
```

## Test

```bash
python run_pipeline.py --cfg_file configs/symnet.yml \
                       --dataset_path PATH/TO/DATASET \
                       --split test \
                       --pipeline.data_generator.test.time_end 800 \
                       --pipeline.data_generator.valid.time_end 800 \
                       --pipeline.test_compute_metric true
```
*Note: The argument ```pipeline.data_generator.test.time_end```, ```pipeline.data_generator.valid.time_end```, and ```pipeline.test_compute_metric``` are examples how to overwrite corresponding entries in the config file.*

The generated test files are stored in the ```pipeline.output_dir``` folder, specified in the config. The output files have a *hdf5* format and can be rendered with the ```utils/draw_sim2d.py``` script.

Rendering of a small sample sequence:
```bash
python utils/draw_sim2d.py PATH/TO/HDF5_FILE OUTPUT/PATH
```

Rendering of individual frames:
```bash
python utils/draw_sim2d.py PATH/TO/HDF5_FILE OUTPUT/PATH \
                           --out_pattern OUTPUT/FRAMES/{frame:04d}.png \
                           --num_frames 800
```

## Validation

```bash
python run_pipeline.py --cfg_file configs/symnet.yml \
                       --dataset_path PATH/TO/DATASET \
                       --split valid
```

## Licenses
Code and scripts are under the MIT license.

Data files are under the CDLA-Permissive-2.0 license.
