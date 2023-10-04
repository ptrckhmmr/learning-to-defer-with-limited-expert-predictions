## Synthetic CIFAR Expert Generation

This repository contains the required code to generate the synthetic expert labels for the CIFAR100 dataset. For a detailed description of the process we refer to the Experiments section of the paper.

### Generating the Synthetic Experts
For generating a synthetic expert for the CIFAR100 dataset with a strength of 60 execute the following command:
```
python generate_synthetic_experts.py --num_classes 20 --strength 60
```
The generated expert labels will be stored in the folder `synthetic_experts/` and added to the `Learning-to-Defer-Algs/artificial_expert_labels` directory.

