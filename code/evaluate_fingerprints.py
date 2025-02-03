import numpy as np
import pandas as pd

import polaris as po
from polaris.hub.client import PolarisHubClient

import matplotlib.pyplot as plt
import seaborn as sns

import os
import pickle
import argparse
from functools import partial
import warnings
warnings.filterwarnings("ignore")

from utils.misc import evaluate_gp
from utils.get_data import get_data


"""
TO DO:
    - Add command line arguments for fingerprint parameters (fp_types, radius, sizes, etc.)
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

DATASET = "biogen/adme-fang-v1"

smiles_train, smiles_test, y_train, y_test = get_data(DATASET)


def write_data(fps, sizes, radius, tol, exp):

    # Instantiate data dicts
    means, vars, mses, pearsons, tlls, gp_params = {}, {}, {}, {}, {}, {}


    # Evaluate GP performance for each fingerprint / size
    for fp in fps:
        for size in sizes:
            key = fp + '-' + str(size)
            mean, var, mse, pearson, tll, params = evaluate_gp(smiles_train,
                                                               smiles_test,
                                                               y_train,
                                                               y_test,
                                                               fp_type=fp,
                                                               sparse=False,
                                                               fpSize=size,
                                                               radius=radius,
                                                               tol=tol,
                                                               max_iters=10000)

            means[key], vars[key], mses[key], pearsons[key], tlls[key], gp_params[key] = mean, var, mse, pearson, tll, params

        # Evaluate GP with SPARSE fingerprint
        key = fp + '-sparse'
        mean, var, mse, pearson, tll, params = evaluate_gp(smiles_train,
                                                           smiles_test,
                                                           y_train,
                                                           y_test,
                                                           fp_type=fp,
                                                           radius=radius,
                                                           tol=tol,
                                                           max_iters=10000)

        means[key], vars[key], mses[key], pearsons[key], tlls[key], gp_params[key] = mean, var, mse, pearson, tll, params


    PATH = f'data/{exp}/'

    # If directory doesn't exist, make it
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    
    # Write data to pickle file
    with open(f'data/{exp}/means.pkl', 'wb') as file:
        pickle.dump(means, file)
    with open(f'data/{exp}/vars.pkl', 'wb') as file:
        pickle.dump(vars, file)
    with open(f'data/{exp}/mses.pkl', 'wb') as file:
        pickle.dump(mses, file)
    with open(f'data/{exp}/pearsons.pkl', 'wb') as file:
        pickle.dump(pearsons, file)
    with open(f'data/{exp}/tlls.pkl', 'wb') as file:
        pickle.dump(tlls, file)
    with open(f'data/{exp}/gp_params.pkl', 'wb') as file:
        pickle.dump(gp_params, file)



def read_data(exp):

    # Read data from pickle files
    with open(f'data/{exp}/means.pkl', 'rb') as file:
        means = pickle.load(file)
    with open(f'data/{exp}/vars.pkl', 'rb') as file:
        vars = pickle.load(file)
    with open(f'data/{exp}/mses.pkl', 'rb') as file:
        mse = pickle.load(file)
    with open(f'data/{exp}/pearsons.pkl', 'rb') as file:
        pearson = pickle.load(file)
    with open(f'data/{exp}/tlls.pkl', 'rb') as file:
        tll = pickle.load(file)
    with open(f'data/{exp}/gp_params.pkl', 'rb') as file:
        gp_params = pickle.load(file)

    return {
        'Means'     : means,
        'Vars'      : vars,
        'MSE'       : mse,
        'Pearson'   : pearson,
        'TLL'       : tll,
        'GP Params' : gp_params
    }

def plot(exp, fps, sizes, data, savefig=False):

    mses, pearsons, tlls, gp_params = [data[key] for key in list(data.keys())[2:]]

    fig1, axes = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(24, 4))

    fig1.suptitle('Results for LogP Regression with Varying Fingerprints', y=1, fontsize=14)

    for i, label in enumerate(fps):
        
        color = sns.color_palette()[i]

        mse, pearson, tll = [], [], []
        for s in sizes:
            key = label + '-' + str(s)
            mse.append(mses[key])
            pearson.append(pearsons[key])
            tll.append(tlls[key])


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

        axes[2].plot(sizes, tll, label=label, lw=.8, color=color)
        axes[2].scatter(2048, tlls[label + '-sparse'], marker='x', s=10)
        axes[2].scatter(sizes, tll, marker='o', s=10, color=color)

        axes[2].legend(loc='lower right', ncols=4, bbox_to_anchor=(-.275, -.2))

    if savefig:
        PATH = f'../figures/fp_comparison/{exp}/{exp}'
        plt.savefig(PATH, bbox_inches='tight')


    # Create scatter plot
    plt.figure(2, figsize=(10, 6))

    params = pd.DataFrame(data=gp_params).rename(index={0: "Amplitude", 1: "Noise"}).round(3)

    params = params.rename(columns=lambda x: x.replace('topological-', 'top-'))
    params = params.rename(columns=lambda x: x.replace('atompair-', 'ap-'))

    X, y = params.iloc[0], params.iloc[1]

    # Get the base fingerprint type for each column
    def get_fp_type(col_name):
        return col_name.split('-')[0]  # This will give us 'ecfp', 'fcfp', etc.

    # Create a color map for the different fingerprint types
    # fp_types = set(get_fp_type(col) for col in params.columns)
    fp_types = ['ecfp', 'fcfp', 'top', 'ap']

    # Plot each fingerprint type with its own color
    for i, fp_type in enumerate(fp_types):
        # Get columns for this fingerprint type
        cols = [col for col in params.columns if get_fp_type(col) == fp_type]
        
        # Get X and y values for these columns
        X_fp = [X[params.columns.get_loc(col)] for col in cols]
        y_fp = [y[params.columns.get_loc(col)] for col in cols]
        
        # Plot with label
        plt.scatter(X_fp, y_fp, color=sns.color_palette()[i], label=fp_type, alpha=0.7)

    plt.xlabel('Amplitude')
    plt.ylabel('Noise')
    plt.grid(True, alpha=0.3)
    plt.title('Noise vs Amplitude for Different Fingerprint Types')
    plt.legend()

    # Add text labels for each point
    for i, (x_val, y_val, col) in enumerate(zip(X, y, params.columns)):
        plt.annotate(col, (x_val, y_val), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8)
        
    if savefig:
        PATH = f'../figures/fp_comparison/{exp}/{exp}_params'
        plt.savefig(PATH, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def main(exp, generate_data=False, make_plots=False, savefig=False, radius=2, tol=1e-3):

    # Fingerprint parameters
    fps = ['ecfp', 'fcfp', 'topological', 'atompair']
    sizes = [512, 1024, 2048]

    if radius == 4:
        fps = ['ecfp', 'fcfp']
    
    if generate_data:
        write_data(fps, sizes, radius, tol, exp)

    if make_plots:
        data = read_data(exp)
        plot(exp, fps, sizes, data, savefig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str)
    parser.add_argument("--generate_data", action="store_true")
    parser.add_argument("--make_plots", action="store_true")
    parser.add_argument("--savefig", action="store_true")
    parser.add_argument('--radius', type=int, choices=[2, 4], default=2)
    parser.add_argument('--tol', type=float, default=1e-3)

    args = parser.parse_args()

    # Figure path must be included
    args = parser.parse_args()
    if args.exp is None:
        parser.error("--exp must be specified")

    main(exp=args.exp,
         generate_data=args.generate_data,
         make_plots=args.make_plots,
         savefig=args.savefig,
         radius=args.radius,
         tol=args.tol)