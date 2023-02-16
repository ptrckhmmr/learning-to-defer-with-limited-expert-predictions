import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
plt.rc('font', family='Times New Roman', size=13)
DATASET = 'nih'
EX_STRENGTH = 4295342357#4295194124 #4295342357
FRAMEWORKS = {'monzannar_sontag': ['a) Mozannar and Sontag (2020)', 50],
              'raghu': ['b) Raghu et al. (2019)', 100],
              'okati': ['c) Okati, De, and Rodriguez (2021)', 100]}
APPROACHES = {'FixMatch': ['FixMatch', 'lightgreen'],
              'CoMatch': ['CoMatch', 'green'],
              'EmbeddingNN_mult': ['Embedding-NN', 'blue'],
              'EmbeddingSVM_mult': ['Embedding-SVM', 'darkblue'],
              'EmbeddingFM_mult': ['Embedding-FixMatch', 'yellow'],
              'EmbeddingCM_mult': ['Embedding-CoMatch', 'orange']}
LABELS = ['4', '8', '12', '20', '40', '100', '500']
SEEDS = [0, 1, 2, 3, 123]
classifier_performance = {4295194124: 81.13,  4295342357: 83.471}
best_ex_performance = {4295342357: 83.89, 4295194124: 85.79}
axes = {'monzannar_sontag': {4295342357: ((92, 100.5)), 4295194124: ((88, 101))},
        'raghu': {4295342357: ((92, 100.5)), 4295194124: ((88, 101))},
        'okati': {4295342357: ((92, 100.5)), 4295194124: ((88, 101))}}

grid = GridSpec(1, 3, left=0.08, right=0.98, top=0.90, bottom=0.35)
fig = plt.figure(figsize=(16, 5))
g = 0
total_results = []
for f, framework in enumerate(FRAMEWORKS.keys()):
    legend = [None]*9
    with open(f'{framework}/results/data/TrueExpert_{EX_STRENGTH}ex_{FRAMEWORKS[framework][1]}epochs_experiment_{DATASET}_results.json', 'r') as file:
        true_ex_results = json.load(file)

    baxes = plt.subplot(grid[g])

    legend[0] = \
    baxes.plot(LABELS, [true_ex_results['accuracy'][0]/true_ex_results['accuracy'][0]*100] * len(LABELS), label='Complete Expert Predictions', color='black',
               linestyle='--')[0]
    legend[1] = \
    baxes.plot(LABELS, [best_ex_performance[EX_STRENGTH]/true_ex_results['accuracy'][0]*100] * len(LABELS), label='Human Expert Alone', color='grey',
               linestyle='dashdot')[0]
    legend[2] = \
    baxes.plot(LABELS, [classifier_performance[EX_STRENGTH]/true_ex_results['accuracy'][0]*100] * len(LABELS), label='Classifier Alone', color='grey', linestyle='dotted')[0]
    total_results.append([framework, EX_STRENGTH, 'True Expert'] + [true_ex_results['accuracy'][0] / 100] * len(LABELS))
    for a, approach in enumerate(APPROACHES.keys()):
        try:
            with open(f'{framework}/results/data/{approach}_{EX_STRENGTH}ex_{FRAMEWORKS[framework][1]}epochs_experiment_{DATASET}_results.json', 'r') as file:
                results = json.load(file)
        except FileNotFoundError:
            print(f'result file {framework}/results/data/{approach}_{EX_STRENGTH}ex_{FRAMEWORKS[framework][1]}epochs_experiment_{DATASET}_results.json not found')
            continue
        acc = {}
        std = {}
        for l in LABELS:
            try:
                acc[l] = results[f'acc_{approach}_{DATASET}_expert{EX_STRENGTH}@{l}'][0]
                std[l] = results[f'acc_{approach}_{DATASET}_expert{EX_STRENGTH}@{l}'][1]
            except KeyError:
                pass
        for key in acc.keys():
            acc[key] = acc[key]/true_ex_results['accuracy'][0]*100
            std[key] = std[key]/true_ex_results['accuracy'][0]*100
        total_results.append([framework, EX_STRENGTH, approach] + list(acc.values()))
        legend[3+a] = baxes.plot(acc.keys(), acc.values(), label=APPROACHES[approach][0], color=APPROACHES[approach][1], marker='o')[0]

        fill_low = [acc[l] - std[l] for l in LABELS]
        fill_up = [acc[l] + std[l] for l in LABELS]

        plt.fill_between(acc.keys(), fill_low, fill_up, alpha=0.1, color=APPROACHES[approach][1])
    if framework == 'monzannar_sontag':
        plt.ylabel(f'NIH Expert {EX_STRENGTH}\n % of System Test Accuracy\n with Complete Expert Predictions', fontsize=14)
    plt.xlabel('Number of Expert Predictions $\mathit{l}$', fontsize=14)
    plt.title(f'{FRAMEWORKS[framework][0]}', fontsize=18)
    plt.minorticks_on()
    plt.grid(visible=True, which='major', alpha=0.2, color='grey', linestyle='-')
    plt.ylim(axes[framework][EX_STRENGTH])
    #baxes.spines['top'].set_visible(False)
    #baxes.spines['right'].set_visible(False)
    if g == 2:
        g += 1
    g += 1
tmp_legend = []
for handle in legend:
    if handle is not None:
        tmp_legend.append(handle)
fig.legend(handles=tmp_legend, loc='lower center', ncol=4, fontsize=14)
plt.savefig(f'plots/results_nih_{EX_STRENGTH}.png', transparent=True)
plt.savefig(f'plots/results_nih_{EX_STRENGTH}.pdf', bbox_inches='tight')
plt.show()

results_df = pd.DataFrame(data=total_results, columns=['framework', 'strength', 'approach'] + LABELS)
results_df.to_csv('results/final_results_nih.csv')