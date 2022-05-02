#!/bin/bash

#python main.py --approach=TrueExpert --epochs=100 --batch=500 --ex_strength=4295342357 --dataset=nih
#python main.py --approach=EmbeddingSVM_bin --epochs=100 --batch=200 --ex_strength=4295342357 --dataset=nih
#4295342357
#4323195249

for labels in 40 80 120 200 400 1000 5000; do
  sbatch -p sdil -n1 -t666 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=CoMatch --epochs=100 --batch=256 --ex_strength=4295342357 --dataset=nih --labels=${labels}"
  sbatch -p sdil -n1 -t666 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingCM_bin --epochs=100 --batch=256 --ex_strength=4295342357 --dataset=nih --labels=${labels}"
  sbatch -p sdil -n1 -t666 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=FixMatch --epochs=100 --batch=256 --ex_strength=4295342357 --dataset=nih --labels=${labels}"
  sbatch -p sdil -n1 -t666 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingFM_bin --epochs=100 --batch=256 --ex_strength=4295342357 --dataset=nih --labels=${labels}"
  sbatch -p sdil -n1 -t666 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingSVM_bin --epochs=100 --batch=256 --ex_strength=4295342357 --dataset=nih --labels=${labels}"
  sbatch -p sdil -n1 -t666 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingNN_bin --epochs=100 --batch=256 --ex_strength=4295342357 --dataset=nih --labels=${labels}"
done