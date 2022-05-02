import matplotlib.pyplot as plt
import json
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, fbeta_score
import numpy as np
import pandas as pd


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


def get_latex_table(acc, std, labels):
    """Get latex table of the results

    :param acc: Accuracies
    :param std: Standard deviations
    :param labels: Labels
    :return: Table as string
    """
    baselines = ['FixMatch', 'CoMatch', 'EmbeddingNN_mult', 'EmbeddingSVM_mult']
    approaches = ['EmbeddingFM_mult', 'EmbeddingCM_mult']
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
                out += ' & \\textbf{'+str(round(100*acc[b][l], 2))+'}'+f'({round(100*std[b][l], 2)})'
            else:
                out += f' & {round(100*acc[b][l], 2)} ({round(100*std[b][l], 2)})'

        out += '\\\\'
    out += '\n \\midrule'
    for a in approaches:
        out += f'\n{a}'
        for i, l in enumerate(labels):
            if a == best[l]:
                out += ' & \\textbf{' + str(round(100 * acc[a][l], 2)) + '}' + f' ({round(100 * std[a][l], 2)})'
            else:
                out += f' & {round(100 * acc[a][l], 2)} ({round(100 * std[a][l], 2)})'
        out += '\\\\'
    out += '\n \\bottomrule'
    return out


strength = 4295342357
APPROACHES = {'FixMatch': ['FixMatch', 'lightgreen'],
              'CoMatch': ['CoMatch', 'green'],
              'EmbeddingNN_mult': ['Embedding-NN', 'blue'],
              'EmbeddingSVM_mult': ['Embedding-SVM', 'darkblue'],
              'EmbeddingFM_mult': ['Embedding-FixMatch', 'yellow'],
              'EmbeddingCM_mult': ['Embedding-CoMatch', 'orange']}
SEEDS = [0, 1, 2, 3, 4]
LABELS = ['4', '8', '12', '20', '40', '100', '500']

data = pd.read_csv('artificial_expert_labels/nih_labels.csv')
data = data[data['Reader ID'] == strength]
data = data.reset_index()
_, test_idx = generate_patient_train_test_split(data)
true_expert = data.loc[test_idx]
#true_expert = data

test_acc = {a: {} for a in APPROACHES.keys()}
mean_acc = {a: {} for a in APPROACHES.keys()}
std_acc = {a: {} for a in APPROACHES.keys()}
test_bacc = {a: {} for a in APPROACHES.keys()}
mean_bacc = {a: {} for a in APPROACHES.keys()}
std_bacc = {a: {} for a in APPROACHES.keys()}

results_emb_svm = pd.read_csv('artificial_expert_labels/emb_svm_label_test_3_nih.csv')
total_results = []
for approach in APPROACHES.keys():
    for labels in LABELS:
        test_acc[approach][labels] = []
        test_bacc[approach][labels] = []
        for seed in SEEDS:
            try:
                predictions_file = f'artificial_expert_labels/{approach}_nih_binary{strength}.{seed}@{labels}_predictions.json'
                with open(predictions_file, 'r') as f:
                    predictions = json.load(f)
                preds = [predictions[img_id] for img_id in true_expert['Image ID']]
                true_preds = 1*(true_expert['Airspace_Opacity_GT_Label']==true_expert['Airspace_Opacity_Expert_Label'])
                test_acc[approach][labels].append(accuracy_score(true_preds, preds))
                test_bacc[approach][labels].append(fbeta_score(true_preds, preds, beta=0.5))
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
plt.ylim(0.5,0.95)
#plt.title(f'Predicted Expert Performance (Strength {strength})', fontsize=14)
plt.minorticks_on()
plt.grid(visible=True, which='major', alpha=0.2, color='grey', linestyle='-')
plt.grid(visible=True, which='minor', alpha=0.1, color='grey', linestyle='-')
#plt.legend(loc='lower right', fontsize=10)
plt.savefig(f"plots/pred_results_nih{strength}.png", transparent=True)
plt.show()

print(get_latex_table(mean_bacc, std_bacc, LABELS))

results_df = pd.DataFrame(data=total_results, columns=['approach'] + LABELS)
results_df.to_csv(f'results/pred_results_nih{strength}.csv')