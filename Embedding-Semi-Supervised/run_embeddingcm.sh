#!/bin/bash
#--n-imgs-per-epoch 32768
for seed in 0 1 2 3 4; do
  for labels in 4 8 12 20 40 100 500; do
    #python Train_embedding_cm.py --exp-dir EmbeddingCM_mult --n-labeled $labels --n-imgs-per-epoch 32768 --seed $seed --ex_strength 4295194124 --batchsize 64 --dataset NIH --n-epoches 25
    sbatch -p sdil -n1 -t918 --gres=gpu:1  --cpus-per-task=20 --wrap "python Train_embedding_cm.py --exp-dir EmbeddingCM_mult --n-labeled $labels --n-imgs-per-epoch 16384 --seed $seed --ex_strength 4295194124 --batchsize 64 --dataset NIH --n-epoches 25"
  done
done

