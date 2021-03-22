import pickle
import matplotlib

import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt

colors = {
    'mbpo': '#1f77b4',
}

# tasks = ['inverted_pendulum', 'hopper', 'walker2d', 'ant', 'cheetah', 'humanoid']
tasks = ['hopper', 'walker2d']
algorithms = ['mbpo']

for task in tasks:
    plt.clf()

    for alg in algorithms:
        print(task, alg)

        ## load results
        fname = '{}_{}.pkl'.format(task, alg)
        data = pickle.load(open(os.path.join('./data/mbpo_original/', fname), 'rb'))

        ## plot trial mean
        plt.plot(data['x'], data['y'], linewidth=1.5, label=alg, c=colors[alg])
        ## plot error bars
        plt.fill_between(data['x'], data['y'] - data['std'], data['y'] + data['std'], color=colors[alg], alpha=0.25)

    plt.legend()

    savepath = './results/{}.png'.format(task)
    plt.savefig(savepath)
