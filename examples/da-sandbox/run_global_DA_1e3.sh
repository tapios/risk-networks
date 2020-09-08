#!/bin/bash

#SBATCH --time=04:00:00                 # walltime
#SBATCH --ntasks=1                      # number of processor cores (i.e. tasks)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -J "global_DA_regularized"
#SBATCH --output=slurm_output/%A_%a.out
#SBATCH --error=slurm_output/%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-7


# create a directory named "slurm_output" and then submit with:
#       sbatch --mail-user=mail@domain.com run_global_regularized.sh


# preface ######################################################################
set -euo pipefail

OUTPUT_DIR="output"
EXP_NAME="global_DA_regularized"


# parameters & constants #######################################################
tested=0
test_budgets=(982 491 392 294 196 98 49 0)  
budget=${test_budgets[${SLURM_ARRAY_TASK_ID}]}

network_size=1e3
user_fraction=1.0

I_min_threshold=0.0
I_max_threshold=1.0

batches_sensors=1
batches_records=8
batches_tests=1
update_test="global"
regularization=0.1

num_cpus=${SLURM_CPUS_PER_TASK}

output_path="${OUTPUT_DIR}/${EXP_NAME}_${budget}"
stdout="${output_path}/stdout"
stderr="${output_path}/stderr"

mkdir -p "${output_path}"


# launch #######################################################################
module load python3/3.8.5
srun python3 backward_forward_assimilation.py \
  >${stdout} 2>${stderr} \
  --network-node-count=${network_size} \
  --user-network-user-fraction=${user_fraction} \
  --constants-output-path=${output_path} \
  --observations-I-budget=${budget} \
  --observations-I-fraction-tested=${tested} \
  --observations-I-min-threshold=${I_min_threshold} \
  --observations-I-max-threshold=${I_max_threshold} \
  --assimilation-batches-sensor=${batches_sensors} \
  --assimilation-batches-record=${batches_records} \
  --assimilation-batches-test=${batches_tests} \
  --assimilation-update-test=${update_test} \
  --assimilation-regularization=${regularization} \
  --parallel-num-cpus=${num_cpus} \
  --parallel-flag
