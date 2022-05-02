## Semi-Supervised Learning Approaches for Generating Artificial Expert Labels
This repository contains the implementations of the semi-supervised learning baselines **FixMatch** and **CoMatch** for generating artificial expert labels. 
These implementations are PyTorch implementations of the <a href="https://arxiv.org/abs/2011.11183">CoMatch paper</a> <a href="https://blog.einstein.ai/comatch-advancing-semi-supervised-learning-with-contrastive-graph-regularization/">[Blog]</a>:
<pre>
@inproceedings{CoMatch,
	title={Semi-supervised Learning with Contrastive Graph Regularization},
	author={Junnan Li and Caiming Xiong and Steven C.H. Hoi},
	booktitle={ICCV},
	year={2021}
}</pre>
and the <a href="https://arxiv.org/abs/2001.07685">FixMatch paper</a>:
<pre>
@inproceedings{FixMatch,
	title = {FixMatch: Simplifying Semi-Supervised Learning with Consistency and Conﬁdence},
	journal = {NeurIPS},
	author = {Sohn, Kihyuk and Berthelot, David and Li, Chun-Liang and Zhang, Zizhao and Carlini, Nicholas and Cubuk, Ekin D and Kurakin, Alex and Zhang, Han and Raffel, Colin},
	year = {2020}
}</pre>

### Requirements:
* PyTorch ≥ 1.4
* pip install tensorboard_logger
* download and extract <a href="https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR-100 dataset</a> into ./data/
* download and extract <a href="https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest">NIH dataset</a> and alter the path to the NIH images in datasets/nih.py

### Train and Generate Artificial Expert Labels
To train the model and generate artificial expert labels for the CIFAR-100 dataset run:
<pre>python Train_CoMatch.py --n-labeled $labels --seed $seed --ex_strength $strength --dataset CIFAR100</pre> 
<pre>python Train_FixMatch.py --n-labeled $labels --seed $seed --ex_strength $strength --dataset CIFAR100</pre> 

To train the model and generate artificial expert labels for the NIH dataset run:
<pre>python Train_CoMatch.py --n-labeled $labels --seed $seed --ex_strength $labeler_id --n-imgs-per-epoch 32768 --dataset NIH</pre> 
<pre>python Train_FixMatch.py --n-labeled $labels --seed $seed --ex_strength $labeler_id --n-imgs-per-epoch 32768 --dataset NIH</pre> 

The generated artificial expert labels will be saved as json file into the *artificial_expert_labels* folder.
