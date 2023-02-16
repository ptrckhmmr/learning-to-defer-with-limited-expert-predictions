## Learning to Defer Algorithms
This repository contains the implementations of the following learning to defer algorithms:
* **Mozannar & Sontag** (<a href="https://proceedings.mlr.press/v119/mozannar20b.html">paper</a>, <a href="https://github.com/clinicalml/learn-to-defer">code</a>)
* **Raghu et al.** (<a href="https://arxiv.org/abs/1903.12220">paper</a>, <a href="https://github.com/clinicalml/learn-to-defer">code</a>)
* **Okati et al.** (<a href="https://arxiv.org/abs/2103.08902">paper</a>, <a href="https://github.com/Networks-Learning/differentiable-learning-under-triage">code</a>)

### Train Learning to Defer Algorithm on Artificial Expert Labels
To train the learning to defer algorithms on **CIFAR-100** using the generated artificial expert labels execute:
<pre>
cd Learning-to-Defer-Algs/
python preprocess_binary_predictions_cifar.py --approach=EmbeddingCM_bin --ex_strength=60
cd $LtD-alg
python main.py --approach=EmbeddingCM_bin --ex_strength=60 --dataset=cifar100
</pre>
where _$LtD-alg_ is the name of the respective learning to defer algorithm to be trained, _approach_ is the name of the approach used
to generate the artificial expert labels, and _ex-strength_ is the strength of the synthetic expert.

To train the learning to defer algorithms on the **NIH dataset** using the generated artificial expert labels execute:
<pre>
mkdir Learning-to-Defer-Algs/_LtD-alg_/experiments/
cp nin_images/checkpoint.pretrain Learning-to-Defer-Algs/_LtD-alg_/experiments/
cd Learning-to-Defer-Algs/LtD-alg
python main.py --approach=EmbeddingCM_mult --ex_strength=4295342357 --dataset=nih
</pre>
where _$LtD-alg_ is the name of the respective learning to defer algorithm to be trained, _approach_ is the name of the approach used
to generate the artificial expert labels, and _ex-strength_ is the labeler-id of the real-world expert.

### Train Learning to Defer Algorithm on Complete Expert Labels
To train the learning to defer algorithms on **CIFAR-100** using a complete set of expert labels execute:
<pre>
cd Learning-to-Defer-Algs/$LtD-alg
python main.py --approach=TrueExpert --ex_strength=60 --dataset=cifar100
</pre>
where _$LtD-alg_ is the name of the learning to defer algorithm to be trained and _expert-strength_ is the strength of the synthetic expert.

To train the learning to defer algorithms on the **NIH dataset** using a complete set of expert labels execute:
<pre>
mkdir Learning-to-Defer-Algs/$LtD-alg/experiments/
cp nin_images/checkpoint.pretrain Learning-to-Defer-Algs/$LtD-alg/experiments/
cd Learning-to-Defer-Algs/$LtD-alg
python main.py --approach=TrueExpert --ex_strength=4295342357 --dataset=nih
</pre>
where _$LtD-alg_ is the name of the learning to defer algorithm to be trained, and _ex-strength_ is the labeler-id of the real-world expert.
