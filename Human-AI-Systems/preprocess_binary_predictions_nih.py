import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics._classification import accuracy_score


def generate_patient_train_test_split(data, seed=1234):
    """Generate train test split based on patient ids

    :param data: Dataframe containing the image ids and patient ids
    :param seed: Random seed
    :return: tuple
        - train_idx: Train indices
        - test_idx: Test indices
    """
    patient_ids = np.unique(data['Patient ID'])
    np.random.seed(seed)
    test_ids = np.random.choice(patient_ids, int(len(patient_ids)*0.2))
    test_idx = []
    train_idx = []
    for i, id in enumerate(data['Patient ID']):
        if id in test_ids:
            test_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, test_idx


def get_nonbin_target(bin, ydata):
    """ Get multiclass targets from binary targets

    :param bin: Binary targets
    :param ydata: Ground truth targets
    :return: Multiclass targets
    """
    np.random.seed(123)
    # create empty arrays for the nonbinary targets
    nonbin = {}

    for img in bin.keys():
        # multiclass target = ground truth target if binary target == 1
        if bin[img] == 1:
            nonbin[img] = int(ydata[ydata['Image ID'] == img]['Airspace_Opacity_GT_Label'].values[0])
        # otherwise draw class from uniform distribution
        else:
            nonbin[img] = int(1-ydata[ydata['Image ID'] == img]['Airspace_Opacity_GT_Label'].values[0])

    return nonbin


def get_bin_target(nonbin, data):
    """Get binary target from multiclass target

    :param nonbin: Multiclass target
    :param data: Dataframe containing the ground-truth labels
    :return: Binary targets
    """
    bin = {}

    for img in nonbin.keys():
        if nonbin[img] == int(data[data['Image ID'] == img]['Airspace_Opacity_GT_Label'].values[0]):
            bin[img] = 1
        else:
            bin[img] = 0

    return bin


def main():
    """Get binary targets from multiclass targets of vice versa

    :return:
    """
    EX_STRENGTH = 4295342357
    NUM_CLASSES = 2
    APPROACH = 'FixMatch'
    labels = [4, 8, 12, 20, 40, 100, 500]
    seeds = [0, 1, 2, 3, 4]
    for s in seeds:
        for l in labels:
            try:
                in_file = f'{APPROACH}_nih_expert{EX_STRENGTH}.{s}@{l}_predictions'
                print(f'preprocess predictions from file {in_file}')
                with open('artificial_expert_labels/'+in_file+'.json', 'r') as f:
                    pred = json.load(f)
            except FileNotFoundError:
                print(f'file {in_file} not found')
                continue

            predictions = pred
            print('transform to multiclass')
            data = pd.read_csv('artificial_expert_labels/nih_labels.csv')
            data = data[data['Reader ID'] == EX_STRENGTH]

            #artificial_expert_labels = get_nonbin_target(artificial_expert_labels, data, NUM_CLASSES)
            predictions = get_bin_target(predictions, data)

            print('Check:', accuracy_score(data['Airspace_Opacity_Correct'], list(predictions.values())))


            out_file = f'{APPROACH}_nih_binary{EX_STRENGTH}.{s}@{l}_predictions'
            with open('artificial_expert_labels/'+out_file+'.json', 'w') as f:
                json.dump(predictions, f)
            print(f'save to {out_file}')


if __name__ == '__main__':
    main()