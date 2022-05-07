## Semi-Supervised Learning Approaches for Generating Artificial Expert Labels
This repository contains the implementations of the semi-supervised learning baselines **FixMatch** and **CoMatch** for generating artificial expert labels. 
These implementations are PyTorch implementations of the <a href="https://arxiv.org/abs/2011.11183">CoMatch paper</a> 
and the <a href="https://arxiv.org/abs/2001.07685">FixMatch paper</a> which can be found in this <a href="https://github.com/salesforce/CoMatch">repository</a>.


### Train and Generate Artificial Expert Labels
To train the model and generate artificial expert labels for the CIFAR-100 dataset run:
<pre>python Train_CoMatch.py --n-labeled $labels --seed $seed --ex_strength $strength --dataset CIFAR100</pre> 
<pre>python Train_FixMatch.py --n-labeled $labels --seed $seed --ex_strength $strength --dataset CIFAR100</pre> 

To train the model and generate artificial expert labels for the NIH dataset run:
<pre>python Train_CoMatch.py --n-labeled $labels --seed $seed --ex_strength $labeler_id --n-imgs-per-epoch 32768 --dataset NIH</pre> 
<pre>python Train_FixMatch.py --n-labeled $labels --seed $seed --ex_strength $labeler_id --n-imgs-per-epoch 32768 --dataset NIH</pre> 

The generated artificial expert labels can be found under `artificial_expert_labels/`. 
For evaluating the artificial expert labels refer to `Human-AI-Systems/`.