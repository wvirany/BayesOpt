import polaris as po
from polaris.hub.client import PolarisHubClient

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import argparse
from functools import partial

from utils import evaluate_gp


"""
TO DO:
    - Add command line arguments for fingeprrint parameters (fp_types, radius, sizes, etc.)
    - Add command line argument for dataset / benchmark
"""


# Set up Seaborn plotting style
sns.set_style("darkgrid",
              {"axes.facecolor": ".95",
               "axes.edgecolor": "#000000",
               "grid.color": "#EBEBE7",
               "font.family": "serif",
               "axes.labelcolor": "#000000",
               "xtick.color": "#000000",
               "ytick.color": "#000000",
               "grid.alpha": 0.4 })
sns.set_palette('muted')


# Login to Polaris
client = PolarisHubClient()
client.login()


# Get data from Polaris benchmark
benchmark = po.load_benchmark("polaris/adme-fang-SOLU-1")

train, test = benchmark.get_train_test_split()

smiles_train = train.X
y_train = train.y
smiles_test = test.X


def generate_data(fps, sizes):

    # Instantiate data dicts
    means, vars, results, mses, pearsons, tlls = {}, {}, {}, {}, {}, {}


    # Evaluate GP performance for each fingerprint / size
    for fp in fps:
        for size in sizes:
            key = fp + '-' + str(size)
            mean, var, tll = evaluate_gp(smiles_train, y_train, smiles_test, fp_type=fp, sparse=False, fpSize=size, tol=1e-3)

            means[key], vars[key], tlls[key] = mean, var, tll
            results[key] = benchmark.evaluate(mean)

        key = fp + '-sparse'
        mean, var, tll = evaluate_gp(smiles_train, y_train, smiles_test, fp_type=fp, radius=4, tol=1e-3)

        means[key], vars[key], tlls[key] = mean, var, tll
        results[key] = benchmark.evaluate(mean)


    # Evaluate MSE and Pearson coefficient metrics
    for key in results.keys():
        mses[key] = results[key].results['Score'][1]
        pearsons[key] = results[key].results['Score'][4]
    
    # Write data to pickle file
    with open('data/means.pkl', 'wb') as file:
        pickle.dump(means, file)
    with open('data/vars.pkl', 'wb') as file:
        pickle.dump(vars, file)
    with open('data/results.pkl', 'wb') as file:
        pickle.dump(results, file)
    with open('data/mses.pkl', 'wb') as file:
        pickle.dump(mses, file)
    with open('data/pearsons.pkl', 'wb') as file:
        pickle.dump(pearsons, file)
    with open('data/tlls.pkl', 'wb') as file:
        pickle.dump(tlls, file)


def read_data():

    # Read data from pickle files
    with open('data/means.pkl', 'rb') as file:
        means = pickle.load(file)
    with open('data/vars.pkl', 'rb') as file:
        vars = pickle.load(file)
    with open('data/results.pkl', 'rb') as file:
        results = pickle.load(file)
    with open('data/mses.pkl', 'rb') as file:
        mse = pickle.load(file)
    with open('data/pearsons.pkl', 'rb') as file:
        pearson = pickle.load(file)
    with open('data/tlls.pkl', 'rb') as file:
        tll = pickle.load(file)

    return {
        'Means'     : means,
        'Vars'      : vars,
        'Results'   : results,
        'MSE'       : mse,
        'Pearson'   : pearson,
        'TLL'       : tll
    }

def plot(fps, sizes, data, savefig=False, filename=None):

    results, mses, pearsons, tlls = [data[key] for key in list(data.keys())[2:]]

    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(24, 4))

    fig.suptitle('Results for LogP Regression with Varying Fingerprints', y=1, fontsize=14)

    for i, label in enumerate(fps):
        
        color = sns.color_palette()[i]

        mse, pearson, n_tll = [], [], []
        for s in sizes:
            key = label + '-' + str(s)
            mse.append(results[key].results['Score'][1])
            pearson.append(results[key].results['Score'][4])
            n_tll.append(tlls[key] / 400)


        # Plot 1: MSE
        axes[0].set_title(r'MSE $\downarrow$')

        axes[0].plot(sizes, mse, label=label, lw=.8, color=color)
        axes[0].scatter(2048, mses[label + '-sparse'], marker='x', s=10)
        axes[0].scatter(sizes, mse, marker='o', s=10, color=color)

        axes[0].set_xticks(sizes)
        axes[0].set_xticklabels(sizes)


        # Plot 2: Pearson Coefficient
        axes[1].set_title(r'Pearson $\uparrow$')

        axes[1].plot(sizes, pearson, label=label, lw=.8, color=color)
        axes[1].scatter(2048, pearsons[label + '-sparse'], marker='x', s=10)
        axes[1].scatter(sizes, pearson, marker='o', s=10, color=color)
            

        # Plot 3: Normalized Test Log-Likelihood
        axes[2].set_title('Normalized Test Log-likelihood')

        axes[2].plot(sizes, n_tll, label=label, lw=.8, color=color)
        axes[2].scatter(2048, tlls[label + '-sparse'] / 400, marker='x', s=10)
        axes[2].scatter(sizes, n_tll, marker='o', s=10, color=color)

        axes[2].legend(loc='lower right', ncols=4, bbox_to_anchor=(-0.275, -0.25))

        if savefig:
            PATH = '../figures/' + filename
            plt.savefig(PATH)


def main(new_data=False, make_plots=False, savefig=False, filename=None):

    # Fingerprint parameters
    fps = ['ecfp', 'fcfp']
    sizes = [512, 1024, 2048]
    
    if new_data:
        generate_data(fps, sizes)

    if make_plots:
        data = read_data()
        plot(fps, sizes, data, savefig, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_data", action="store_true")
    parser.add_argument("--make_plots", action="store_true")
    parser.add_argument("--savefig", action="store_true")
    parser.add_argument('--filename', type=str)

    args = parser.parse_args()

    # Figure path must be included
    args = parser.parse_args()
    if args.savefig and args.filename is None:
        parser.error("--filename is required when --savefig is set")

    main(new_data=args.new_data, make_plots=args.make_plots, savefig=args.savefig, filename=args.filename)