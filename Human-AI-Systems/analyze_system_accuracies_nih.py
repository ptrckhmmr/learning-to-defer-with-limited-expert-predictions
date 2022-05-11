import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from brokenaxes import brokenaxes
import pandas as pd

DATASET = 'nih'
EX_STRENGTH = 4295342357
FRAMEWORKS = {'monzannar_sontag': ['a) Monzannar and Sontag', 50],
              'raghu': ['b) Raghu et al.', 100],
              'okati': ['c) Okati et al.', 100]}
APPROACHES = {'FixMatch': ['FixMatch', 'lightgreen'],
              'CoMatch': ['CoMatch', 'green'],
              'EmbeddingNN_mult': ['Embedding-NN', 'blue'],
              'EmbeddingSVM_mult': ['Embedding-SVM', 'darkblue'],
              'EmbeddingFM_mult': ['Embedding-FixMatch (ours)', 'yellow'],
              'EmbeddingCM_mult': ['Embedding-CoMatch (ours)', 'orange']}
LABELS = ['4', '8', '12', '20', '40', '100', '500']
SEEDS = [0, 1, 2, 3, 123]
best_ex_performance = {60: 0.6792, 90: 0.9262, 95: 0.9613, 4295342357: 0.8389}
axes = {'monzannar_sontag': {60: ((0.67,0.69),(0.75,0.87)), 90: ((0.77, 0.78), (0.89, 0.97)), 4295342357: ((0.795, 0.885))},
        'raghu': {60: ((0.67,0.69),(0.77, 0.79),(0.82,0.90)), 90: ((0.77, 0.78), (0.90, 0.97)), 4295342357: ((0.831, 0.885))},
        'okati': {60: ((0.67, 0.69), (0.75, 0.86)), 90: ((0.77, 0.78), (0.85, 0.96)), 4295342357: ((0.815, 0.885))}}

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
    baxes.plot(LABELS, [true_ex_results['accuracy'][0] / 100] * len(LABELS), label='Complete Expert Labels', color='black',
               linestyle='--')[0]
    legend[1] = \
    baxes.plot(LABELS, [best_ex_performance[EX_STRENGTH]] * len(LABELS), label='Human Expert Alone', color='grey',
               linestyle='dashdot')[0]
    legend[2] = \
    baxes.plot(LABELS, [0.83471] * len(LABELS), label='Classifier Alone', color='grey', linestyle='dotted')[0]
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
            acc[key] = acc[key]/100
            std[key] = std[key]/100
        total_results.append([framework, EX_STRENGTH, approach] + list(acc.values()))
        legend[3+a] = baxes.plot(acc.keys(), acc.values(), label=APPROACHES[approach][0], color=APPROACHES[approach][1], marker='o')[0]

        fill_low = [acc[l] - std[l] for l in LABELS]
        fill_up = [acc[l] + std[l] for l in LABELS]

        plt.fill_between(acc.keys(), fill_low, fill_up, alpha=0.1, color=APPROACHES[approach][1])
    if framework == 'monzannar_sontag':
        plt.ylabel(f'NIH Expert {EX_STRENGTH}\n % of System Test Accuracy\n with Complete Expert Labels\n', fontsize=12)
    plt.xlabel('\nNumber of Expert Labels $\mathit{l}$', fontsize=12)
    plt.title(f'{FRAMEWORKS[framework][0]}', fontsize=15)
    plt.minorticks_on()
    plt.grid(visible=True, which='major', alpha=0.2, color='grey', linestyle='-')
    plt.ylim(axes[framework][EX_STRENGTH])
    baxes.spines['top'].set_visible(False)
    baxes.spines['right'].set_visible(False)
    if g == 2:
        g += 1
    g += 1
tmp_legend = []
for handle in legend:
    if handle is not None:
        tmp_legend.append(handle)
fig.legend(handles=tmp_legend, loc='lower center', ncol=4, fontsize=12)
plt.savefig('plots/results_nih.png', transparent=True)
plt.savefig('plots/results_nih.pdf', bbox_inches='tight')
plt.show()

results_df = pd.DataFrame(data=total_results, columns=['framework', 'strength', 'approach'] + LABELS)
results_df.to_csv('results/final_results_nih.csv')