#!/bin/bash

pip install dm-tree

mkdir ${1}/WaterRamps
for file in metadata.json train.tfrecord valid.tfrecord test.tfrecord
do
wget -O "${1}/WaterRamps/${file}" "https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/WaterRamps/${file}"
done

for split in train valid test
do
python utils/tfrecord_msgpack.py --data_path "${1}/WaterRamps" --out_path "${1}/WaterRamps" --split ${split}
done