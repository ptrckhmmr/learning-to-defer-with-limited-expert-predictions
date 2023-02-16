#!/bin/bash

for seed in 0 1 2 3 4; do
  for labels in 4 8 12 20 40 100 500; do
    sbatch -p sdil -n1 -t1700 --gres=gpu:1 --cpus-per-task=20 --wrap "python Train_FixMatch.py --n-labeled $labels --seed $seed --n-imgs-per-epoch 32768  --ex_strength 4295194124 --batchsize 10 --dataset NIH --n-epoches 25"
  done
done