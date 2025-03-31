import numpy as np

import tanimoto_gp

import matplotlib.pyplot as plt
import seaborn as sns

import os
import pickle
from pathlib import Path
import argparse


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


def main(n_train, target):

    pickle_dir = Path(f"results/dockstring-regression/{target}")
    
    print(target)

    # Load results
    for file_path in pickle_dir.iterdir():
        with open(file_path, "rb") as f:
            data = pickle.load(f)

            print(f"{file_path.name}: {data['R2']}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--target", type=str, default='PARP1')

    args = parser.parse_args()

    main(n_train=args.n_train, target=args.target)