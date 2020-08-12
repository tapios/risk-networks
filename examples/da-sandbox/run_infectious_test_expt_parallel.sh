#!/bin/bash

#SBATCH --time=24:00:00                 # walltime
#SBATCH --ntasks=1                      # number of processor cores (i.e. tasks)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --mem=48G                       
#SBATCH -J "I_per_day_test"
#SBATCH --output=output/slurm_%j.out
#SBATCH --error=output/slurm_%j.err  
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-7

##################################
# Infectiousness test experiment #
##################################
# set: fractions_test = % tested of I per day
# set: output_path = ${OUTPUT_DIR}/expt_name_${tested}
# ENSURE you have selected the parallel master equations import from _master_eqns_init.py
# submit with: sbatch run_infectious_test_expt_parallel.sh

# preface ######################################################################
set -euo pipefail

OUTPUT_DIR="output"
EXP_NAME="1e4_SD_long"


# parameters & constants #######################################################
# by fraction
#fraction_tested=(1.0 0.5 0.1 0.05 0.04 0.03 0.02 0.01 0.005 0.0) #Make sure no. expts agrees with size of array. and no commas.
#tested=${fractions_tested[${SLURM_ARRAY_TASK_ID}]}
#output_path="${OUTPUT_DIR}/${EXP_NAME}_${tested}"

# by number
#test_budgets=(0 4)
#test_budgets=(982 491 98 49 39 29 19 9 4 0)  
test_budgets=(982 491 392 294 196 98 49 0)  
budget=${test_budgets[${SLURM_ARRAY_TASK_ID}]}
output_path="${OUTPUT_DIR}/${EXP_NAME}_${budget}"


#other params 
network_size=1e4
I_min_threshold=0.0
I_max_threshold=1.0
user_fraction=1.0
batches=40
parflag=True
num_cores=16
stdout="${output_path}/stdout"
stderr="${output_path}/stderr"

mkdir -p "${output_path}"


# launch #######################################################################
#module load python3/3.7.0
python3 backward_forward_assimilation.py \
  --user-network-user-fraction=${user_fraction} \
  --constants-output-path=${output_path} \
  --observations-I-budget=${budget} \
  --observations-I-min-threshold=${I_min_threshold} \
  --observations-I-max-threshold=${I_max_threshold} \
  --network-node-count=${network_size} \
  --assimilation-batches=${batches} \
  --parallel-flag=${parflag} \
  --num-cores=${num_cores} \
  >${stdout} 2>${stderr}




