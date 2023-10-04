import os
import json
import argparse

from src.CIFAR100_Expert import CIFAR100_Expert
import src.preprocess_data as prep


def main():
    parser = argparse.ArgumentParser(description='Synthetic Expert Label Generation')
    parser.add_argument('--strength', type=int, default=60, help='Strength of the synthetic expert')
    parser.add_argument('--binary', type=bool, default=False, help='Binary labels for the synthetic expert')
    parser.add_argument('--num_classes', type=int, default=20, help='Number of classes in the dataset')
    parser.add_argument('--per_s', type=float, default=1.0, help='Probability for a correct expert label for the expert\'s strengths')
    parser.add_argument('--per_w', type=float, default=0.0, help='Probability for a correct expert label for the expert\'s weaknesses')
    args = parser.parse_args()

    # generate expert of strength X
    expert = CIFAR100_Expert(args.num_classes, args.strength, args.per_s, args.per_w)

    train_data, test_data = prep.get_train_test_data()

    true_ex_labels = {'train': expert.generate_expert_labels(train_data.targets, binary=args.binary).tolist(),
                      'test': expert.generate_expert_labels(test_data.targets, binary=args.binary).tolist()}

    os.makedirs('synthetic_experts', exist_ok=True)
    with open(f'synthetic_experts/TrueEx_cifar100_{args.strength}_labels.json', 'w') as f:
        json.dump(true_ex_labels, f)
    artificial_ex_labels_path = os.getcwd()[:-len('Synthetic_CIFAR_Expert')]+f'Learning-to-Defer-Algs/artificial_expert_labels'
    os.makedirs(artificial_ex_labels_path, exist_ok=True)
    with open(f'{artificial_ex_labels_path}/TrueEx_cifar100_{args.strength}_labels.json', 'w') as f:
        json.dump(true_ex_labels, f)


if __name__ == '__main__':
    main()
