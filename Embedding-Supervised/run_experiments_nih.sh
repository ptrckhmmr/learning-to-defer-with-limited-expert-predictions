#!/bin/bash
for seed in 0 1 2 3 4; do
  for labels in 4 8 12 20 40 100 500; do
    sbatch -p sdil -n1 -t500 --gres=gpu:1  --cpus-per-task=20 --wrap "python train_nn_ex_model.py --n_labeled ${labels} --seed ${seed} --ex_strength 4295194124 --dataset nih --emb_model resnet18 --binary False"
    #sbatch -p sdil -n1 -t100 --gres=gpu:1  --cpus-per-task=20 --wrap "python train_svm_ex_model.py --n_labeled ${labels} --seed ${seed} --ex_strength 4295194124 --dataset nih --emb_model resnet18 --binary False"
  done
done