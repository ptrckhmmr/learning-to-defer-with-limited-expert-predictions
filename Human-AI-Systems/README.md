## Human-AI Systems
This repository contains the implementations of the following human-AI systems:
* **Mozannar & Sontag** (<a href="https://proceedings.mlr.press/v119/mozannar20b.html">paper</a>, <a href="https://github.com/clinicalml/learn-to-defer">code</a>)
* **Raghu et al.** (<a href="https://arxiv.org/abs/1903.12220">paper</a>, <a href="https://github.com/clinicalml/learn-to-defer">code</a>)
* **Okati et al.** (<a href="https://arxiv.org/abs/2103.08902">paper</a>, <a href="https://github.com/Networks-Learning/differentiable-learning-under-triage">code</a>)

### Train Human-AI Systems on Artificial Expert Labels
To train the human-machine systems on CIFAR-100 using the generated artificial expert labels execute:
<pre>
python preprocess_binary_predictions_cifar.py --approach=EmbeddingCM_bin --ex_strength=60
cd system-name
python main.py --approach=EmbeddingCM_bin --ex_strength=60 --dataset=cifar100
</pre>
where _system-name_ is the name of the human-AI system to be trained, _approach_ is the name of the approach used
to generate the artificial expert labels, and _expert-strength_ is the strength of the synthetic expert.

To train the human-machine systems on the NIH dataset using the generated artificial expert labels execute:
<pre>
cd system-name
cp trained-resnet18 experiments/
python main.py --approach=EmbeddingCM_mult --ex_strength=4295342357 --dataset=nih
</pre>
where _system-name_ is the name of the human-AI system to be trained, _approach-name_ is the name of the approach used 
to generate the artificial expert labels, _expert-strength_ is the labeler-id of the real-world expert, and
_trained-resnet18_ are the weights of the trained resnet18 embedding model.

### Train Human-AI Systems on Complete Expert Labels
To train the human-machine systems on CIFAR-100 using a complete set of expert labels execute:
<pre>
cd system-name
python main.py --approach=TrueExpert --ex_strength=60 --dataset=cifar100
</pre>
where _system-name_ is the name of the human-AI system to be trained and _expert-strength_ is the strength of the synthetic expert.

To train the human-machine systems on the NIH dataset using complete set of expert labels execute:
<pre>
cd system-name
cp trained-resnet18 experiments/
python main.py --approach=TrueExpert --ex_strength=4295342357 --dataset=nih
</pre>
where _system-name_ is the name of the human-AI system to be trained, _expert-strength_ is the labeler-id of the real-world expert, and
_trained-resnet18_ are the weights of the trained resnet18 embedding model..

### Evaluate Results
For evaluating the accuracy of the artificial expert labels execute:
* `analyze_artex_labels_cifar.py` (for CIFAR-100) 
* `analyze_artex_labels_nih.py` (for NIH)

For evaluating the system accuracies of the human-AI systems execute:
* `analyze_system_accuracies_cifar.py` (for CIFAR-100) 
* `analyze_system_accuracies_nih.py` (for NIH)

The generated graphs are saved to `plots/`.

