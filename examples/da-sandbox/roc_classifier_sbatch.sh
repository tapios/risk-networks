#!/bin/bash

#SBATCH --time=12:00:00                 # walltime
#SBATCH --ntasks=1                      # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=4G               # 24G is needed for 10k full user base
#SBATCH -J "risk_networks_accuracy"
#SBATCH --output=output/slurm_%j.out
#SBATCH --error=output/slurm_%j.err  
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-7

# submit with: sbatch roc_classifier_sbatch.sh

# preface ######################################################################
set -euo pipefail

OUTPUT_DIR="output"


# parameters & constants #######################################################
fractions_tested=(1.0 0.5 0.1 0.05 0.04 0.03 0.02 0.01) #Make sure # expts agrees with size of array.
network_size=1e3
I_min_threshold=0.0
user_fraction=1.0
tested=${fractions_tested[${SLURM_ARRAY_TASK_ID}]}
output_path="${OUTPUT_DIR}/sum_roc_itest_${tested}"
stdout="${output_path}/stdout"
stderr="${output_path}/stderr"

mkdir -p "${output_path}"


# launch #######################################################################
module load python3/3.7.0
python3 roc_classifier.py \
  --user-network-user-fraction=${user_fraction} \
  --constants-output-path=${output_path} \
  --observations-I-fraction-tested=${tested} \
  --observations-I-min-threshold=${I_min_threshold} \
  --network-node-count=${network_size} \
  >${stdout} 2>${stderr}


