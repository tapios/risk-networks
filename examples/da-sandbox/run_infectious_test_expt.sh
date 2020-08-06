#!/bin/bash

#SBATCH --time=48:00:00                 # walltime
#SBATCH --ntasks=1                      # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=24G 
#SBATCH -J "I_expt"
#SBATCH --output=output/slurm_%j.out
#SBATCH --error=output/slurm_%j.err  
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-9

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
EXP_NAME="test_serial"

# parameters & constants #######################################################
# by fraction
#fraction_tested=(1.0 0.5 0.1 0.05 0.04 0.03 0.02 0.01 0.005 0.0) #Make sure no. expts agrees with size of array. and no commas.
#tested=${fractions_tested[${SLURM_ARRAY_TASK_ID}]}
#output_path="${OUTPUT_DIR}/${EXP_NAME}_${tested}"

# by number
#for 1e3
test_budgets=(982 491 98 49 39 29 19 9 4 0)  
#for 1e4
#test_budgets=(982 491 392 294 196 98 49 0)  
budget=${test_budgets[${SLURM_ARRAY_TASK_ID}]}
output_path="${OUTPUT_DIR}/${EXP_NAME}_${budget}"


#parsed parameters 
network_size=1e3
I_min_threshold=0.0
I_max_threshold=1.0
user_fraction=1.0
batches=4
parflag=False
stdout="${output_path}/stdout"
stderr="${output_path}/stderr"

mkdir -p "${output_path}"


# launch #######################################################################
python3 backward_forward_assimilation.py \
  --user-network-user-fraction=${user_fraction} \
  --constants-output-path=${output_path} \
  --observations-I-budget=${budget} \
  --observations-I-min-threshold=${I_min_threshold} \
  --observations-I-max-threshold=${I_max_threshold} \
  --network-node-count=${network_size} \
  --assimilation-batches=${batches}
  --parallel-flag=${parflag}
  >${stdout} 2>${stderr}

#  --observations-I-fraction-tested=${tested} \
