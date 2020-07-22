#!/bin/bash

#SBATCH --time=00:30:00                # walltime
#SBATCH --nodes=1                      # number of nodes
#SBATCH --ntasks=1                     # number of processor cores (i.e. tasks)
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --output=run.log

srun accuracy_estimation.py
