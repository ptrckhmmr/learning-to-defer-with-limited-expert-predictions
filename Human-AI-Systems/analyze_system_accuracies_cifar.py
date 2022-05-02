import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from brokenaxes import brokenaxes
import pandas as pd

DATASET = 'cifar100'
EX_STRENGTH = [60, 90]
FRAMEWORKS = {'monzannar_sontag': ['a) Monzannar and Sontag', 50],
              'raghu': ['b) Raghu et al.', 200],
              'okati': ['c) Okati et al.', 100]}
APPROACHES = {'FixMatch': ['FixMatch', 'lightgreen'],
              'CoMatch': ['CoMatch', 'green'],
              'EmbeddingNN_bin': ['Embedding-NN', 'blue'],
              'EmbeddingSVM_bin': ['Embedding-SVM', 'darkblue'],
              'EmbeddingFM_bin': ['Embedding-FixMatch', 'yellow'],
              'EmbeddingCM_bin': ['Embedding-CoMatch', 'orange']}
LABELS = ['40', '80', '120', '200', '400', '1000', '5000']
SEEDS = [0, 1, 2, 3, 123]
best_ex_performance = {60: 0.6792, 90: 0.9262, 95: 0.9613, 4295342357: 0.8389}
axes = {'monzannar_sontag': {60: ((0.67,0.69),(0.75,0.87)), 90: ((0.77, 0.78), (0.895, 0.97))},
        'raghu': {60: ((0.67,0.69),(0.77, 0.79),(0.82,0.90)), 90: ((0.77, 0.78), (0.905, 0.97))},
        'okati': {60: ((0.67, 0.69), (0.75, 0.86)), 90: ((0.77, 0.78), (0.85, 0.96))}}

grid = GridSpec(2, 4, left=0.08, right=1.3, top=0.95, bottom=0.2)
fig = plt.figure(figsize=(16, 8))
g = 0
total_results = []
for s, strength in enumerate(EX_STRENGTH):
    for f, framework in enumerate(FRAMEWORKS.keys()):
        legend = [None]*9
        with open(f'{framework}/results/data/TrueExpert_{strength}ex_{FRAMEWORKS[framework][1]}epochs_experiment_{DATASET}_results.json', 'r') as file:
            true_ex_results = json.load(file)

        #axs[s,f].figure(figsize=(8, 5))
        baxes = brokenaxes(ylims=axes[framework][strength], hspace=0.05, subplot_spec=grid[g], d=0.006)

        legend[0] = baxes.plot(LABELS, [true_ex_results['accuracy'][0]/100]*len(LABELS), label='Complete Expert Labels', color='black', linestyle='--')[0][0]
        legend[1] = baxes.plot(LABELS, [best_ex_performance[strength]]*len(LABELS), label='Human Expert Alone', color='grey', linestyle='dashdot')[0][0]
        legend[2] = baxes.plot(LABELS, [0.7785]*len(LABELS), label='Classifier Alone', color='grey', linestyle='dotted')[0][0]
        total_results.append([framework, strength, 'True Expert'] + [true_ex_results['accuracy'][0]/100]*len(LABELS))
        for a, approach in enumerate(APPROACHES.keys()):
            try:
                with open(f'{framework}/results/data/{approach}_{strength}ex_{FRAMEWORKS[framework][1]}epochs_experiment_{DATASET}_results.json', 'r') as file:
                    results = json.load(file)
            except FileNotFoundError:
                print(f'result file {framework}/results/data/{approach}_{strength}ex_{FRAMEWORKS[framework][1]}epochs_experiment_{DATASET}_results.json not found')
                continue
            acc = {}
            std = {}
            for l in LABELS:
                try:
                    acc[l] = results[f'acc_{approach}_{DATASET}_expert{strength}@{l}'][0]
                    std[l] = results[f'acc_{approach}_{DATASET}_expert{strength}@{l}'][1]
                except KeyError:
                    pass
            for key in acc.keys():
                acc[key] = acc[key]/100
                std[key] = std[key]/100
            total_results.append([framework, strength, approach] + list(acc.values()))
            legend[3+a] = baxes.plot(acc.keys(), acc.values(), label=APPROACHES[approach][0], color=APPROACHES[approach][1], marker='o')[0][0]

            fill_low = [acc[l] - std[l] for l in LABELS]
            fill_up = [acc[l] + std[l] for l in LABELS]

            baxes.fill_between(acc.keys(), fill_low, fill_up, alpha=0.1, color=APPROACHES[approach][1])
        if framework == 'monzannar_sontag':
            baxes.set_ylabel(f'Synthetic Expert (Strength {strength}) \nSystem Test Accuracy\n', fontsize=12)
        if strength == 90:
            baxes.set_xlabel('\nNumber of Expert Labels ($\mathit{l}$)', fontsize=12)
        if strength == 60:
            baxes.set_title(f'{FRAMEWORKS[framework][0]}', fontsize=15)
        baxes.minorticks_on()
        baxes.grid(visible=True, which='major', alpha=0.2, color='grey', linestyle='-')

        #plt.grid(visible=True, which='minor', alpha=0.1, color='grey', linestyle='-')
        #fig.add_subplot(grid[g])
        if g == 2:
            g += 1
        g += 1

fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=12)
plt.savefig('plots/results_cifar.png', transparent=True)
plt.savefig('plots/results_cifar.pdf', bbox_inches='tight')
plt.show()

results_df = pd.DataFrame(data=total_results, columns=['framework', 'strength', 'approach'] + LABELS)
results_df.to_csv('results/final_results_cifar.csv')