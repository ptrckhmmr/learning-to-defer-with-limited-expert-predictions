## Embedding Semi-Supervised Learning Approaches for Generating Artificial Expert Labels
This repository contains the implementations of the embedding semi-supervised learning baselines *Embedding-FixMatch* and *Embedding-CoMatch* for generating artificial expert labels. 
These implementations use the code from the PyTorch implementations of <a href="https://arxiv.org/abs/2011.11183">CoMatch</a> 
and the <a href="https://arxiv.org/abs/2001.07685">FixMatch</a> which can be found in this <a href="https://github.com/salesforce/CoMatch">repository</a>.

### Train Embedding Model
To train the embedding model for the CIFAR-100 dataset run:
<pre>python train_emb_model.py --dataset cifar100 --model efficientnetb1 --num_classes 20 --lr 0.1</pre> 

To train the embedding model for the NIH dataset run:
<pre>python train_emb_model.py --dataset nih --model resnet18 --num_classes 2 --lr 0.001</pre> 

### Train Expert Model and Generate Artificial Expert Labels
To train the expert model and generate artificial expert labels for the CIFAR-100 dataset run:
<pre>python Train_embedding_cm.py --n-labeled 120 --ex_strength 60 --dataset CIFAR100</pre> 
<pre>python Train_embedding_fm.py --n-labeled 120 --ex_strength 60 --dataset CIFAR100</pre> 

To train the expert model and generate artificial expert labels for the NIH dataset run:
<pre>python Train_embedding_cm.py --n-labeled 12 --n-imgs-per-epoch 32768 --dataset NIH</pre> 
<pre>python Train_embedding_fm.py --n-labeled 12 --n-imgs-per-epoch 32768 --dataset NIH</pre> 

The generated artificial expert labels can be found under `artificial_expert_labels/`. 
For evaluating the artificial expert labels refer to `Human-AI-Systems/analyze_artex_labels_cifar.py` and  `Human-AI-Systems/analyze_artex_labels_nih.py`.
