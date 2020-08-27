#!/bin/bash

#SBATCH --time=48:00:00                 # walltime
#SBATCH --ntasks=1                      # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=24G
#SBATCH -J "kin_ens"
#SBATCH --output=output/slurm_%j.out
#SBATCH --error=output/slurm_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-10

##################################
# Infectiousness test experiment #
##################################
# set: fractions_test = % tested of I per day
# set: output_path = ${OUTPUT_DIR}/expt_name_${tested}
# ENSURE you have the non-parallel import from _master_eqns_init
# submit with: sbatch run_infectious_test_expt.sh

# preface ######################################################################
set -euo pipefail

OUTPUT_DIR="output"
EXP_NAME="kinetic_ensemble"

# parameters & constants #######################################################
# by fraction
#fraction_tested=(0.5 0.4 0.3 0.2 0.1 0.05 0.01 0.0) #Make sure no. expts agrees with size of array. and no commas.
#tested=${fractions_tested[${SLURM_ARRAY_TASK_ID}]}
#output_path="${OUTPUT_DIR}/${EXP_NAME}_${tested}"

# by number
#for 1e3
#test_budgets=9
#for 1e4
#test_budgets=(982 491 392 294 196 98 49 0)
#budget=${test_budgets[${SLURM_ARRAY_TASK_ID}]}
#output_path="${OUTPUT_DIR}/${EXP_NAME}_${budget}"

#parsed parameters
parflag=False
stdout="${output_path}/stdout"
stderr="${output_path}/stderr"
array_num=$(printf "%04d" ${SLURM_ARRAY_TASK_ID})
epidemic_storage="epidemic_storage_$array_num.pkl"

mkdir -p "${output_path}"

# launch #######################################################################
python3 run_kinetic_model.py \
  --constants-output-path=${output_path} \
  --kinetic_seed=${SLURM_ARRAY_TASK_ID} \
  --epidemic-storage-name=${epidemic_storage} \
  --network-node-count=${network_size} \
  --parallel-flag=${parflag} \
  >${stdout} 2>${stderr}
