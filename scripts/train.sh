#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 14
#SBATCH --mail-user=lukas.prantl@tum.de
#SBATCH --mail-type=END

cd deep-eng.-fluids
source activate tf
python -W ignore run_pipeline.py --cfg_file configs/symnet.yml --dataset_path ${WORK}/datasets/data_complex_02/ --pipeline.version 2d --split train --restart
