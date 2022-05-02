## Embedding-Supervised Learning Approaches for Generating Artificial Expert Labels
This repository contains the implementations of the embedding-supervised learning baselines **Embedding-NN** and **Embedding-SVM** for generating artificial expert labels. 

### Requirements:
* PyTorch ≥ 1.4
* download and extract <a href="https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR-100 dataset</a> into ./data/
* download and extract <a href="https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest">NIH dataset</a> and alter the path to the NIH images in datasets/nih.py

### Train Embedding Model
To train the embedding model for the CIFAR-100 dataset run:
<pre>python train_emb_model.py --dataset cifar100 --model efficientnetb1 --num_classes 20 --lr 0.1</pre> 

To train the embedding model for the NIH dataset run:
<pre>python train_emb_model.py --dataset nih --model resnet18 --num_classes 2 --lr 0.001</pre> 

### Train Expert Model and Generate Artificial Expert Labels
To train the expert model and generate artificial expert labels for the CIFAR-100 dataset run:
<pre>python train_nn_ex_model.py --n-labeled $labels --seed $seed --ex_strength $strength --dataset cifar100 --emb_model efficientnetb1 --binary True</pre> 
<pre>python train_svm_ex_model.py --n-labeled $labels --seed $seed --ex_strength $strength --dataset cifar100 --emb_model efficientnetb1 --binary True</pre> 

To train the expert model and generate artificial expert labels for the NIH dataset run:
<pre>python train_nn_ex_model.py --n-labeled $labels --seed $seed --ex_strength $strength --dataset nih --emb_model resnet18 --binary False</pre> 
<pre>python train_SVM_ex_model.py --n-labeled $labels --seed $seed --ex_strength $strength --dataset nih --emb_model resnet18 --binary False</pre> 

The generated artificial expert labels will be saved as json file into the *artificial_expert_labels* folder.
