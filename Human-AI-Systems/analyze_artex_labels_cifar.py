import matplotlib.pyplot as plt
import json
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, confusion_matrix, fbeta_score
import numpy as np
import pandas as pd


def get_latex_table(acc, std, labels):
    """Get latex table of the results

    :param acc: Accuracies
    :param std: Standard deviations
    :param labels: Labels
    :return: Table as string
    """
    baselines = ['FixMatch', 'CoMatch', 'EmbeddingNN_bin', 'EmbeddingSVM_bin']
    approaches = ['EmbeddingFM_bin', 'EmbeddingCM_bin']
    best = {}
    for l in labels:
        top = 0
        for key in acc.keys():
            if acc[key][l] > top:
                best[l] = key
                top = acc[key][l]

    print('**********************START-TABLE**********************')
    out = ''
    for b in baselines:
        out += f'\n{b}'
        for i, l in enumerate(labels):
            if b == best[l]:
                out += ' & \\textbf{'+str(round(100*acc[b][l], 2))+'}'+f' ({round(100*std[b][l], 2)})'
            else:
                out += f' & {round(100*acc[b][l], 2)} ({round(100*std[b][l], 2)})'

        out += '\\\\'
    out += '\n \\midrule'
    for a in approaches:
        out += f'\n{a}'
        for i, l in enumerate(labels):
            if a == best[l]:
                out += ' & \\textbf{' + str(round(100 * acc[a][l], 2)) + '}' + f'({round(100 * std[a][l], 2)})'
            else:
                out += f' & {round(100 * acc[a][l], 2)} ({round(100 * std[a][l], 2)})'
        out += '\\\\'
    out += '\n \\bottomrule'
    return out

strength = 90
APPROACHES = {'FixMatch': ['FixMatch', 'lightgreen'],
              'CoMatch': ['CoMatch', 'green'],
              'EmbeddingNN_bin': ['Embedding-NN', 'blue'],
              'EmbeddingSVM_bin': ['Embedding-SVM', 'darkblue'],
              'EmbeddingFM_bin': ['Embedding-FixMatch', 'yellow'],
              'EmbeddingCM_bin': ['Embedding-CoMatch', 'orange']}

SEEDS = [0, 1, 2, 3, 4]
LABELS = ['40', '80', '120', '200', '400', '1000', '5000']

with open(f'artificial_expert_labels/TrueEx_bin_cifar100_{strength}_labels.json') as t:
    true_expert = json.load(t)

test_acc = {a: {} for a in APPROACHES}
mean_acc = {a: {} for a in APPROACHES}
std_acc = {a: {} for a in APPROACHES}
test_bacc = {a: {} for a in APPROACHES}
mean_bacc = {a: {} for a in APPROACHES}
std_bacc = {a: {} for a in APPROACHES}

results_emb_svm = pd.read_csv(f'artificial_expert_labels/emb_svm_label_test_large.csv')
total_results = []
for approach in APPROACHES.keys():
    for labels in LABELS:
        test_acc[approach][labels] = []
        test_bacc[approach][labels] = []
        for seed in SEEDS:
            if approach in ['EmbeddingNN_bin'] and seed == 4: seed = 123
            try:
                predictions_file = f'artificial_expert_labels/{approach}_cifar100_binary{strength}.{seed}@{labels}_predictions.json'
                if approach == 'BinaryMeanTeacher':
                    predictions_file = f'artificial_expert_labels/{approach}_cifar100_binary_{strength}.{seed}@{labels}_predictions.json'
                with open(predictions_file, 'r') as f:
                    predictions = json.load(f)
                cm = confusion_matrix(true_expert['test'], predictions['test'])
                test_acc[approach][labels].append(accuracy_score(true_expert['test'], predictions['test']))
                test_bacc[approach][labels].append(fbeta_score(true_expert['test'], predictions['test'], beta=0.5))
            except FileNotFoundError:
                print(f'{predictions_file} not found')
                pass
        mean_acc[approach][labels] = np.mean(test_acc[approach][labels])
        std_acc[approach][labels] = np.std(test_acc[approach][labels], ddof=1)/np.sqrt(np.size(test_acc[approach][labels]))
        mean_bacc[approach][labels] = np.mean(test_bacc[approach][labels])
        std_bacc[approach][labels] = np.std(test_bacc[approach][labels], ddof=1)/np.sqrt(np.size(test_bacc[approach][labels]))
    total_results.append([approach] + list(mean_bacc[approach].values()))

plt.figure(figsize=(8, 4.5))
plt.style.use('seaborn-colorblind')
for approach in APPROACHES:
    plt.plot(mean_bacc[approach].keys(), mean_bacc[approach].values(), label=APPROACHES[approach][0], color=APPROACHES[approach][1], marker='o')
    fill_low = [mean_bacc[approach][l] - std_bacc[approach][l] for l in LABELS]
    fill_up = [mean_bacc[approach][l] + std_bacc[approach][l] for l in LABELS]
    plt.fill_between(std_bacc[approach].keys(), fill_low, fill_up, alpha=0.1, color=APPROACHES[approach][1])
plt.xlabel('Number of Expert Labels', fontsize=14)
plt.ylabel('F0.5-Score', fontsize=14)
#plt.title(f'Predicted Expert Performance (Strength {strength})', fontsize=14)
plt.minorticks_on()
plt.grid(visible=True, which='major', alpha=0.2, color='grey', linestyle='-')
plt.grid(visible=True, which='minor', alpha=0.1, color='grey', linestyle='-')
#plt.legend(loc='lower right', fontsize=10)
plt.savefig(f"plots/pred_results_cifar{strength}.png", transparent=True)
plt.show()

print(get_latex_table(mean_bacc, std_bacc, LABELS))

results_df = pd.DataFrame(data=total_results, columns=['approach'] + LABELS)
results_df.to_csv(f'results/pred_results_cifar{strength}.csv')