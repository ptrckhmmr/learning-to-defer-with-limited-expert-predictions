import json
import os
import numpy as np
import pickle
from absl import flags
from absl import app
from sklearn.metrics._classification import accuracy_score

FLAGS = flags.FLAGS

def unpickle(file):
    """Function to open the files using pickle

    :param file: File to be loaded
    :return: Loaded file as dictionary
    """
    with open(file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='latin1')
    return myDict


def load_coarse_targets(wkdir):
    """Load CIFAR100 fine targets

    :param wkdir: Working directory
    :return: tuple (trainData, testData, metaData)
        - trainData['fine_labels'] - fine labels for training data
        - testData['fine_labels'] - fine labels for test data
    """
    trainData = unpickle(wkdir + '/data/cifar-100-python/train')
    testData = unpickle(wkdir + '/data/cifar-100-python/test')

    return trainData['coarse_labels'], testData['coarse_labels']


def get_nonbin_target(bin, y, num_classes):
    """Get multiclass targets from binary targets

    :param bin: Binary targets
    :param y: Ground truth targets
    :param num_classes: Number of classes
    :return: Multiclass targets
    """
    np.random.seed(123)
    # create empty arrays for the nonbinary targets
    nonbin = np.zeros(len(y), dtype=int)

    for i in range(len(y)):
        # multiclass target = ground truth target if binary target == 1
        if bin[i] == 1:
            nonbin[i] = y[i]
        # otherwise draw class from uniform distribution
        else:
            nonbin[i] = int(np.random.uniform(0, num_classes))

    return nonbin.tolist()

def main():
    """Get multiclass expert label from binary expert label

    :return:
    """
    EX_STRENGTH = FLAGS.ex_strength
    NUM_CLASSES = 20
    APPROACH = FLAGS.approach
    labels = [40, 80, 120, 200, 400, 1000, 5000]
    seeds = [0, 1, 2, 3, 4]
    for s in seeds:
        for l in labels:
            try:
                in_file = f'{APPROACH}_cifar100_binary{EX_STRENGTH}.{s}@{l}_predictions'
                print(f'preprocess artificial_expert_labels from file {in_file}')
                with open('artificial_expert_labels/'+in_file+'.json', 'r') as f:
                    bmt_pred = json.load(f)
            except FileNotFoundError:
                print(f'file {in_file} not found')
                continue

            predictions = bmt_pred
            print('transform to multiclass')
            train_gt_targets, test_gt_targets = load_coarse_targets(os.getcwd()+'/hemmer')

            predictions['train'] = get_nonbin_target(predictions['train'], train_gt_targets, NUM_CLASSES)
            predictions['test'] = get_nonbin_target(predictions['test'], test_gt_targets, NUM_CLASSES)

            print('Check train:', accuracy_score(train_gt_targets, predictions['train']))
            print('Check test:', accuracy_score(test_gt_targets, predictions['test']))

            out_file = f'{APPROACH}_cifar100_expert{EX_STRENGTH}.{s}@{l}_predictions'
            with open('artificial_expert_labels/'+out_file+'.json', 'w') as f:
                json.dump(predictions, f)
            print(f'save to {out_file}')


if __name__ == '__main__':
    flags.DEFINE_string('approach', 'EmbeddingCM_bin', 'Approach for predicting the expert labels')
    flags.DEFINE_integer('ex_strength', 60, 'Expert Strength')
    app.run(main)