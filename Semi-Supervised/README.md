## Semi-Supervised Learning Approaches for Generating Artificial Expert Labels
This repository contains the implementations of the semi-supervised learning baselines *FixMatch* and *CoMatch* for generating artificial expert labels. 
These baselines are PyTorch implementations of the <a href="https://arxiv.org/abs/2011.11183">CoMatch paper</a> 
and the <a href="https://arxiv.org/abs/2001.07685">FixMatch paper</a> which are available in this <a href="https://github.com/salesforce/CoMatch">repository</a>.


### Train and Generate Artificial Expert Labels
To train the model and generate artificial expert labels for the **CIFAR-100** dataset download and extract the
<a href="https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR-100 dataset</a> into `./data/` and run:
<pre>python Train_CoMatch.py --n-labeled 120 --ex_strength 60 --dataset CIFAR100</pre> 
<pre>python Train_FixMatch.py --n-labeled 120 --ex_strength 60 --dataset CIFAR100</pre>
where _n-labeled_ is the number of given expert labels, and _ex-strength_ is the strength of the synthetic expert.

To train the model and generate artificial expert labels for the **NIH dataset** run:
<pre>python Train_CoMatch.py --n-labeled 12 --ex_strength 4295342357 --n-imgs-per-epoch 32768 --dataset NIH</pre>
<pre>python Train_FixMatch.py --n-labeled 12 --ex_strength 4295342357 --n-imgs-per-epoch 32768 --dataset NIH</pre>
where _n-labeled_ is the number of given expert labels, and _ex-strength_ is the labeler-id of the human expert.

The generated artificial expert labels can be found under `artificial_expert_labels/`. 
