import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
plt.rc('font', family='Times New Roman', size=13)

DATASET = 'cifar100'
EX_STRENGTH = [60, 90]
FRAMEWORKS = {'mozannar_sontag': ['a) Mozannar and Sontag (2020)', 50],
              'raghu': ['b) Raghu et al. (2019)', 200],
              'okati': ['c) Okati, De, and Rodriguez (2021)', 100]}
APPROACHES = {'FixMatch': ['FixMatch', 'lightgreen'],
              'CoMatch': ['CoMatch', 'green'],
              'EmbeddingNN_bin': ['Embedding-NN', 'blue'],
              'EmbeddingSVM_bin': ['Embedding-SVM', 'darkblue'],
              'EmbeddingFM_bin': ['Embedding-FixMatch', 'yellow'],
              'EmbeddingCM_bin': ['Embedding-CoMatch', 'orange']}
LABELS = ['40', '80', '120', '200', '400', '1000', '5000']
SEEDS = [0, 1, 2, 3, 123]
best_ex_performance = {60: 67.92, 90: 92.62, 95: 96.13, 4295342357: 83.89}
axes = {60: (87, 101), 90: (89, 101)}
grid = GridSpec(2, 4, left=0.09, right=1.3, top=0.95, bottom=0.2)
fig = plt.figure(figsize=(16, 8))
g = 0
total_results = []
legend = [None] * 9
for s, strength in enumerate(EX_STRENGTH):
    for f, framework in enumerate(FRAMEWORKS.keys()):

        with open(f'{framework}/results/data/TrueExpert_{strength}ex_{FRAMEWORKS[framework][1]}epochs_experiment_{DATASET}_results.json', 'r') as file:
            true_ex_results = json.load(file)

        baxes = plt.subplot(grid[g])
        legend[0] = baxes.plot(LABELS, [true_ex_results['accuracy'][0]/true_ex_results['accuracy'][0]*100]*len(LABELS), label='Complete Expert Predictions', color='black', linestyle='--')[0]
        if strength == 90:
            legend[1] = baxes.plot(LABELS, [best_ex_performance[strength]/true_ex_results['accuracy'][0]*100]*len(LABELS), label='Human Expert Alone', color='grey', linestyle='dashdot')[0]
        else:
            legend[2] = baxes.plot(LABELS, [77.85/true_ex_results['accuracy'][0]*100]*len(LABELS), label='Classifier Alone', color='grey', linestyle='dotted')[0]
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
                acc[key] = acc[key]/true_ex_results['accuracy'][0]*100
                std[key] = std[key]/true_ex_results['accuracy'][0]*100
            total_results.append([framework, strength, approach] + list(acc.values()))
            legend[3+a] = baxes.plot(acc.keys(), acc.values(), label=APPROACHES[approach][0], color=APPROACHES[approach][1], marker='o')[0]

            fill_low = [acc[l] - std[l] for l in LABELS]
            fill_up = [acc[l] + std[l] for l in LABELS]

            baxes.fill_between(acc.keys(), fill_low, fill_up, alpha=0.1, color=APPROACHES[approach][1])
        if framework == 'mozannar_sontag':
            plt.ylabel(r'Synthetic Expert $H_{'+str(strength)+'}$ \n % of System Test Accuracy\n with Complete Expert Predictions', fontsize=14)
        if strength == 90:
            plt.xlabel('Number of Expert Predictions $\mathit{l}$', fontsize=14)

        if strength == 60:
            plt.title(f'{FRAMEWORKS[framework][0]}', fontsize=18)


        plt.minorticks_on()
        plt.grid(visible=True, which='major', alpha=0.2, color='grey', linestyle='-')
        plt.ylim(axes[strength])

        if g == 2:
            g+=1
        g += 1

fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=14)
plt.savefig('plots/results_cifar_relative.png', transparent=True)
plt.savefig('plots/results_cifar_relative.pdf', bbox_inches='tight')
plt.show()

results_df = pd.DataFrame(data=total_results, columns=['framework', 'strength', 'approach'] + LABELS)
results_df.to_csv('results/final_results_cifar_relative.csv')