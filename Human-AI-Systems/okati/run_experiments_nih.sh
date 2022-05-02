#!/bin/bash

#python main.py --approach=TrueExpert --epochs=50 --batch=500 --ex_strength=4295342357 --dataset=nih
sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingCM_mult --epochs=100 --batch=512 --ex_strength=4295342357 --dataset=nih"
sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingFM_mult --epochs=100 --batch=512 --ex_strength=4295342357 --dataset=nih"
sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingSVM_mult --epochs=100 --batch=512 --ex_strength=4295342357 --dataset=nih"
sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingNN_mult --epochs=100 --batch=512 --ex_strength=4295342357 --dataset=nih"
sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=FixMatch --epochs=100 --batch=512 --ex_strength=4295342357 --dataset=nih"
sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=CoMatch --epochs=100 --batch=512 --ex_strength=4295342357 --dataset=nih"
#python main.py --approach=EmbeddingNN_mult --epochs=100 --batch=500 --ex_strength=4295342357 --dataset=nih

#for labels in 4 8 12 20 40 100 500; do
  #sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingCM_mult --epochs=100 --batch=512 --ex_strength=4295342357 --dataset=nih --labels=${labels}"
  #sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=FixMatch --epochs=100 --batch=512 --ex_strength=4295342357 --dataset=nih --labels=${labels}"
#  sleep 5
#done