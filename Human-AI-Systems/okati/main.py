import torch.utils.data
from absl import flags
from absl import app
import numpy as np
import json
import random
import os
import sys

from src.train import run_expert, run_classifier
from src.experts import Cifar100Expert, NihExpert

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FLAGS = flags.FLAGS


def main(argv):
    #############################################
    NUM_CLASSES = 20 if FLAGS.dataset == 'cifar100' else 2
    TRAIN_BATCH_SIZE = FLAGS.batch
    TEST_BATCH_SIZE = 2*FLAGS.batch
    EPOCHS = FLAGS.epochs
    WKDIR = os.getcwd()
    NUM_EXPERTS = 1
    EX_STRENGTH = FLAGS.ex_strength
    APPROACH = FLAGS.approach
    DATASET = FLAGS.dataset
    BINARY = FLAGS.binary
    LABELS = [FLAGS.labels]
    SEEDS = [4, 0, 1, 2, 3]
    #############################################
    if LABELS[0] is None:
        if DATASET == 'nih':
            LABELS = [4, 8, 12, 20, 40, 100, 500]
        else:
            LABELS = [40, 80, 120, 200, 400, 1000, 5000]

    if 'BinaryMeanTeacher' in APPROACH or 'EmbeddingBMT' in APPROACH or DATASET == 'nih':
        SEEDS[0] = 4
    # generate args dictionary
    args = {'approach': APPROACH+'_probs', 'ex_strength': EX_STRENGTH, 'num_experts': NUM_EXPERTS, 'binary': BINARY}
    # get filename for true expert preds
    true_ex = f'TrueEx_{DATASET}'
    true_ex_preds = f'{true_ex}_{EX_STRENGTH}_labels'

    # run human-AI collaboration with true expert
    if APPROACH == 'TrueExpert':
        args['approach'] = true_ex
        accuracies = []
        coverages = {'cov': [], 'cov_by_class': []}
        if DATASET == 'cifar100':
            true_expert = Cifar100Expert(pred_dir=WKDIR, pred=true_ex_preds, true_pred=true_ex_preds)
        elif DATASET == 'nih':
            true_expert = NihExpert(pred_dir=WKDIR, pred=true_ex_preds, true_pred=true_ex_preds)
        else:
            print(f'Dataset {DATASET} not implemented')
            sys.exit()
        # get ture expert prediction function
        true_expert_fns = true_expert.predict
        # run human-AI collaboration for the true expert
        for seed in [12345]:
            print(f'Seed: {seed}')
            print("-" * 40)
            model_classifier = run_classifier(args, true_expert_fns, 100, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, seed,
                                              NUM_CLASSES)
            best_metrics = run_expert(args, model_classifier, true_expert_fns, EPOCHS, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE,
                                      seed, NUM_CLASSES)
            # record metrics
            accuracies.append(best_metrics['system accuracy'])
            coverages['cov'].append(best_metrics['coverage'])
            coverages['cov_by_class'].append(best_metrics['cov per class'])

    # run human-AI collaboration with the predicted expert
    else:
        pred_keys = [f'{APPROACH}_{DATASET}_expert{EX_STRENGTH}@{l}' for l in LABELS]
        # initiate arrays and dicts for the accuracies and coverages
        accuracies = {p: [] for p in pred_keys}
        coverages = {p: {'cov': [], 'cov_by_class': []} for p in pred_keys}
        # run human-AI collaboration for different seeds
        for seed in SEEDS:
            print(f'Seed: {seed}')
            print("-" * 40)
            np.random.seed(seed)
            random.seed(seed)
            # get filenames for artificial_expert_labels
            if seed == 123 or ('EmbeddingBMT' in APPROACH and seed == 4):
                predictions = [f'{APPROACH}_{DATASET}_expert{EX_STRENGTH}@{l}_predictions' for l in LABELS]
            elif seed > 4:
                predictions = [f'{APPROACH}_{DATASET}_expert{EX_STRENGTH}.{4}@{l}_predictions' for l in LABELS]
            else:
                predictions = [f'{APPROACH}_{DATASET}_expert{EX_STRENGTH}.{seed}@{l}_predictions' for l in LABELS]
            keys = [f'{APPROACH}_{DATASET}_expert{EX_STRENGTH}@{l}' for l in LABELS]

            pred_expert = {}
            pred_expert_fns = {}
            for i, p in enumerate(predictions):
                try:
                    if DATASET == 'cifar100':
                        pred_expert[p] = Cifar100Expert(pred_dir=WKDIR, pred=p, true_pred=true_ex_preds)

                    elif DATASET == 'nih':
                        pred_expert[p] = NihExpert(pred_dir=WKDIR, pred=p, true_pred=true_ex_preds)
                    pred_expert_fns[p] = pred_expert[p].predict
                    print('Get performance of collaboration with the pred expert ' + p + ':')
                    args['approach'] = p
                    model_classifier = run_classifier(args, pred_expert_fns[p], 100, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE,
                                                      seed,
                                                      NUM_CLASSES)
                    best_metrics = run_expert(args, model_classifier, pred_expert_fns[p], EPOCHS, TRAIN_BATCH_SIZE,
                                              TEST_BATCH_SIZE,
                                              seed, NUM_CLASSES)
                    accuracies[keys[i]].append(best_metrics['system accuracy'])
                    coverages[keys[i]]['cov'].append(best_metrics['coverage'])
                    coverages[keys[i]]['cov_by_class'].append(best_metrics['cov per class'])
                except FileNotFoundError:
                    print(f'Predictions file for {p} not found')
                    pass
    # save results of human-AI collaboration framework
    experiment_data = {}
    if APPROACH == 'TrueExpert':
        experiment_data['accuracy'] = [np.mean(accuracies), np.std(accuracies)]
        experiment_data['coverage'] = [[np.mean(coverages['cov']), np.std(coverages['cov'])],
                                       [np.mean(coverages['cov_by_class'], axis=0).tolist(),
                                        np.std(coverages['cov_by_class'], axis=0).tolist()]]
    else:
        for p in pred_keys:
            try:
                if len(accuracies[p]) > 0:
                    experiment_data['acc_' + p] = [np.mean(accuracies[p]),  np.std(accuracies[p])]
                    experiment_data['cov_' + p] = [[np.mean(coverages[p]['cov']), np.std(coverages[p]['cov'])],
                                                  [np.mean(coverages[p]['cov_by_class'], axis=0).tolist(),
                                                   np.std(coverages[p]['cov_by_class'], axis=0).tolist()]]
            except KeyError:
                pass
    if not os.path.exists(WKDIR + '/results'):
        os.makedirs(WKDIR + '/results')
    with open(f'{WKDIR}/results/{APPROACH}_{EX_STRENGTH}ex_{EPOCHS}epochs_experiment_{DATASET}_results.json', 'w') as fp:
        json.dump(experiment_data, fp)


if __name__ == '__main__':
    flags.DEFINE_string('approach', 'TrueExpert', 'Approach for predicting the expert labels')
    flags.DEFINE_string('dataset', 'cifar100', 'Dataset')
    flags.DEFINE_integer('epochs', 100, 'Number of epochs to train')
    flags.DEFINE_integer('batch', 512, 'Batchsize')
    flags.DEFINE_integer('ex_strength', 60, 'Expert Strength')
    flags.DEFINE_integer('labels', None, 'Number of Expert Labels')
    flags.DEFINE_boolean('binary', False, 'Use binary expert labels')
    app.run(main)