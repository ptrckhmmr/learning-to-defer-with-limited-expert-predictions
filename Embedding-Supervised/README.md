## Embedding Supervised Learning Approaches for Generating Artificial Expert Labels
This repository contains the implementations of the embedding-supervised learning baselines *Embedding-NN* and *Embedding-SVM* for generating artificial expert labels. 

### Train Embedding Model
To train the embedding model for the CIFAR-100 dataset run:
<pre>python train_emb_model.py --dataset cifar100 --model efficientnetb1 --num_classes 20 --lr 0.1</pre> 

To train the embedding model for the NIH dataset run:
<pre>python train_emb_model.py --dataset nih --model resnet18 --num_classes 2 --lr 0.001</pre> 

### Train Expert Model and Generate Artificial Expert Labels
To train the expert model and generate artificial expert labels for the CIFAR-100 dataset run:
<pre>python train_nn_ex_model.py --n-labeled 120 --ex_strength 60 --dataset cifar100 --emb_model efficientnetb1 --binary True</pre> 
<pre>python train_svm_ex_model.py --n-labeled 120 --ex_strength 60 --dataset cifar100 --emb_model efficientnetb1 --binary True</pre> 

To train the expert model and generate artificial expert labels for the NIH dataset run:
<pre>python train_nn_ex_model.py --n-labeled 12 --dataset nih --emb_model resnet18 --binary False</pre> 
<pre>python train_SVM_ex_model.py --n-labeled 12 --dataset nih --emb_model resnet18 --binary False</pre> 

The generated artificial expert labels can be found under `artificial_expert_labels/`. 
For evaluating the artificial expert labels refer to `Human-AI-Systems/analyze_artex_labels_cifar.py` and `Human-AI-Systems/analyze_artex_labels_nih.py`.
