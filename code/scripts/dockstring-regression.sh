#!/bin/bash

#SBATCH --partition=amilan
#SBATCH --job-name=dockstring-regression-F2-baseline-compressed-count
#SBATCH --output=logs/dockstring-regression/F2-baseline-compressed-count.out
#SBATCH --time=8:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=waltervirany@gmail.com

module purge
module load python
module load anaconda
conda activate tanimoto-gp

# Add JAX CPU optimization settings
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=16"

# Run the Python script with appropriate parameters
python dockstring-regression.py --target "F2"