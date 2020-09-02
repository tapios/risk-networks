#!/bin/bash

#SBATCH --time=24:00:00                 # walltime
#SBATCH --ntasks=1                      # number of processor cores (i.e. tasks)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                       
#SBATCH -J "I_per_day_test"
#SBATCH --output=output/slurm_%j.out
#SBATCH --error=output/slurm_%j.err  
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-0

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
EXP_NAME="1e4_RDT_DAsweep_3"

output_path="${OUTPUT_DIR}/${EXP_NAME}"
# parameters & constants #######################################################
#fraction_tested=(0.5 0.4 0.3 0.2 0.1 0.05 0.01 0.0) #Make sure no. expts agrees with size of array. and no commas.
#tested=${fractions_tested[${SLURM_ARRAY_TASK_ID}]}
#output_path="${OUTPUT_DIR}/${EXP_NAME}_${tested}"
#batches_test=2
tested=0


# by number
#test_budgets=(490 981 2451)  
#budget=${test_budgets[${SLURM_ARRAY_TASK_ID}]}
#batches_tests=(1 2 5)
#batches_test=${batches_tests[${SLURM_ARRAY_TASK_ID}]}
#output_path="${OUTPUT_DIR}/${EXP_NAME}_${budget}"
batches_test=5
budget=2451

#by sensor wearers
#sensor_wearers=(2451 4903 9807)
#batches_sensors=(5 10 20)
#batches_sensor=${batches_sensors[${SLURM_ARRAY_TASK_ID}]}
#wearers=${sensor_wearers[${SLURM_ARRAY_TASK_ID}]}
#output_path="${OUTPUT_DIR}/${EXP_NAME}_${wearers}"
wearers=0
batches_sensor=1

#parsed parameters 
network_size=1e4
I_min_threshold=0.0
I_max_threshold=1.0
user_fraction=1.0
batches_records=40
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
  --observations-I-fraction-tested=${tested} \
  --observations-sensor-wearers=${wearers} \
  --observations-I-min-threshold=${I_min_threshold} \
  --observations-I-max-threshold=${I_max_threshold} \
  --network-node-count=${network_size} \
  --assimilation-batches-sensor=${batches_sensor} \
  --assimilation-batches-test=${batches_test} \
  --assimilation-batches-record=${batches_records} \
  --parallel-flag \
  --num-cores=${num_cores} \
  >${stdout} 2>${stderr}




