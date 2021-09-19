#!/bin/bash

python train.dist.apex.py --nodes=1 --gpus=8 --nr=0  --train-file="data/DIV2K_x2.h5" --eval-file="data/Set5_x2.h5" --outputs-dir="outputs-apex" --scale=2 --num-features=64 --growth-rate=64 --num-blocks=16 --num-layers=8  --lr=1e-4 --batch-size=32  --patch-size=32  --num-epochs=1000   --num-workers=8  --seed=123
