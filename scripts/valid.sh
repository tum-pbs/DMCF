#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mail-user=lukas.prantl@tum.de
#SBATCH --mail-type=END

cd deep-eng.-fluids
source activate tf

export SCRIPT=symnet
export version=2d
python -W ignore run_pipeline.py --cfg_file configs/${SCRIPT}.yml --dataset_path ${WORK}/datasets/valid/complex2/ --split valid --pipeline.version ${version} --pipeline.data_generator.valid.time_end 800 --pipeline.data_generator.valid.random_start 0
python -W ignore run_pipeline.py --cfg_file configs/${SCRIPT}.yml --dataset_path ${WORK}/datasets/valid/tank/ --split valid --pipeline.version ${version} --pipeline.data_generator.valid.time_end 400 --pipeline.data_generator.valid.random_start 0
python -W ignore run_pipeline.py --cfg_file configs/${SCRIPT}.yml --dataset_path ${WORK}/datasets/valid/momentum/ --split valid --pipeline.version ${version} --pipeline.data_generator.valid.time_end 200 --pipeline.data_generator.valid.random_start 0
python -W ignore run_pipeline.py --cfg_file configs/${SCRIPT}.yml --dataset_path ${WORK}/datasets/valid/momentum_g/ --split valid --pipeline.version ${version} --pipeline.data_generator.valid.time_end 200 --pipeline.data_generator.valid.random_start 0


# python -W ignore run_pipeline.py --cfg_file configs/${SCRIPT}.yml --dataset_path ${WORK}/datasets/valid/complex2/ --split valid --pipeline.data_generator.valid.time_end 400 --pipeline.data_generator.valid.random_start 400
# python -W ignore run_pipeline.py --cfg_file configs/${SCRIPT}.yml --dataset_path ${WORK}/datasets/valid/tank/ --split valid --pipeline.data_generator.valid.time_end 400 --pipeline.data_generator.valid.random_start 200
# python -W ignore run_pipeline.py --cfg_file configs/${SCRIPT}.yml --dataset_path ${WORK}/datasets/valid/momentum/ --split valid --pipeline.data_generator.valid.time_end 200 --pipeline.data_generator.valid.random_start 0
# #python -W ignore run_pipeline.py --cfg_file configs/${SCRIPT}.yml --dataset_path ${WORK}/datasets/valid/momentum_g/ --split valid --pipeline.data_generator.valid.time_end 150 --pipeline.data_generator.valid.random_start 0

