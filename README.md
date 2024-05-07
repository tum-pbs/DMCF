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
- Install cmake: ```sudo apt install cmake```
- Update pip: ```pip install --upgrade pip```
- Install requirements: ```pip install -r requirements.txt```
- Tensorpack DataFlow ```pip install --upgrade git+https://github.com/tensorpack/dataflow.git```

Optional: 
- Build FPS/EMD module ```cd utils; make; cd ..```
- Install skia for visualization: ```python -m pip install skia-python```

## Datasets

- *WBC-SPH*: https://dataserv.ub.tum.de/index.php/s/m1693614
- *Liquid3d* ([source](https://github.com/isl-org/DeepLagrangianFluids)): https://drive.google.com/file/d/1b3OjeXnsvwUAeUq2Z0lcrX7j9U7zLO07
- *WaterRamps* ([source](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate)): ```bash download_waterramps.sh PATH/TO/OUTPUT_DIR```

## Pretrained Models:

The pretrained models are in the ```checkpoints``` subfolder.
Run a pretrained mode by setting the path to the checkpoint with the ```ckpt_path``` argument.
For example:
```bash
python run_pipeline.py --cfg_file configs/WBC-SPH.yml \
                       --dataset_path PATH/TO/DATASET \
                       --ckpt_path checkpoints/WBC-SPH/ckpt \
                       --split test
```

## Training

Simple 1D test run (data will be generated):
```bash
python run_pipeline.py --cfg_file configs/column/hrnet.yml \
                       --split train
```

Run with 2D pipeline:
```bash
python run_pipeline.py --cfg_file configs/WBC-SPH.yml \
                       --dataset_path PATH/TO/DATASET \
                       --split train
```

## Test

```bash
python run_pipeline.py --cfg_file configs/WBC-SPH.yml \
                       --dataset_path PATH/TO/DATASET \
                       --split test \
                       --pipeline.data_generator.test.time_end 800 \
                       --pipeline.data_generator.valid.time_end 800 \
                       --pipeline.data_generator.valid.random_start 0 \
                       --pipeline.test_compute_metric true
```
*Note: The argument ```pipeline.data_generator.test.time_end```, ```pipeline.data_generator.valid.time_end```, ```pipeline.data_generator.valid.random_start```, and ```pipeline.test_compute_metric``` are examples how to overwrite corresponding entries in the config file.*

The ```...time_end``` parameter account for the number of frames used for inference and evaluation. We used a value of *3200* for the *WBC-SPH* data set, *600* for *WaterRamps*, and *200* for *Liquid3d*.
The generated test files are stored in the ```pipeline.output_dir``` folder, specified in the config file. The output files have a *hdf5* format and can be rendered with the ```utils/draw_sim2d.py``` script.

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
python run_pipeline.py --cfg_file configs/WBC-SPH.yml \
                       --dataset_path PATH/TO/DATASET \
                       --split valid
```

## FAQ and Clarifications

### How was the training data generated with SPlisHSPlasH? 
The data was generated based on the code of previous work (https://github.com/isl-org/DeepLagrangianFluids). The required code is in the 'datasets' folder. 'create_data.sh' is the shell script to run the data generation.

### Why not replace all the CConv with the ASCC?
The reason for this is that the antisymmetry is a severe restriction. This makes the problem much more difficult for the neural network to solve. It is sufficient to place the ASCC only at the end to make the network antisymmetric. Even with this, it was very difficult to adjust the mesh to the current state. On the other hand, the network can generalized much better. The antisymmetric layer can also be seen as a kind of mapping into a reduced antisymmetric space. 
The paper has a short paragraph with an example (Standing Liquid) in the Result section, which briefly discusses this.

### What is the *Maximum Density* in the evaluation?
The *Maximum Density* value is the relative error between the maximum density of the fluid and the maximum density of the ground truth, where a value closer to 0 is preferable (Equation 15 in the paper). We use this as a heuristic for the compressibility of the fluid, which can lead to high pressure and thus instability in the simulation. Apart from that, please note that the values in Table 2 in the paper are not the raw error values but relative accuracy values as described in Figure 6. I.e. a value of 1 corresponds to the error of our final method, while small values represent a lower relative accuracy and thus larger error. A value of 0.5, for example, would mean half the accuracy and double the error. We chose this format to relate the error to the final method, which we felt was important in an ablation study, and to normalise the error evaluation for better visualisation in the graph.

## Licenses
Code and scripts are under the MIT license.

Data files are under the CDLA-Permissive-2.0 license.
