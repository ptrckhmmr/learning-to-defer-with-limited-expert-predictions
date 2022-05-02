#!/bin/bash
sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=CoMatch --epochs=100 --batch=512 --ex_strength=90"
sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=FixMatch --epochs=100 --batch=512 --ex_strength=90"
sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingFM_bin --epochs=100 --batch=512 --ex_strength=90"
sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingCM_bin --epochs=100 --batch=512 --ex_strength=90"
sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingNN_bin --epochs=100 --batch=512 --ex_strength=90"
sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingSVM_bin --epochs=100 --batch=512 --ex_strength=90"
#python main.py --approach=EmbeddingSVM_bin --epochs=200 --batch=1028 --ex_strength=60
#sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=TrueExpert --epochs=100 --batch=512 --ex_strength=90"
#sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingNN_bin --epochs=100 --batch=512 --ex_strength=60"
#sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingSVM_bin --epochs=100 --batch=512 --ex_strength=60"
#for labels in 40 80 120 200 400 1000 5000; do
  #sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=CoMatch --epochs=100 --batch=512 --ex_strength=90 --labels=${labels}"
  #sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=FixMatch --epochs=100 --batch=512 --ex_strength=90 --labels=${labels}"
  #sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingFM_bin --epochs=100 --batch=512 --ex_strength=90 --labels=${labels}"
  #sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingCM_bin --epochs=100 --batch=512 --ex_strength=90 --labels=${labels}"
  #sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingNN_bin --epochs=100 --batch=512 --ex_strength=90 --labels=${labels}"
  #sbatch -p sdil -n1 -t999 --gres=gpu:1  --cpus-per-task=20 --wrap "python main.py --approach=EmbeddingSVM_bin --epochs=100 --batch=512 --ex_strength=90 --labels=${labels}"
#done