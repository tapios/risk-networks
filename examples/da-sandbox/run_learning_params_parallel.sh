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
EXP_NAME="1e4_sensor_itest_rand_learn_transmission_rate"


# parameters & constants #######################################################
#fraction_tested=(0.5 0.4 0.3 0.2 0.1 0.05 0.01 0.0) #Make sure no. expts agrees with size of array. and no commas.
#tested=${fractions_tested[${SLURM_ARRAY_TASK_ID}]}
#output_path="${OUTPUT_DIR}/${EXP_NAME}_${tested}"

# by number
#for 1e3
#test_budgets=9  
#for 1e4
test_budgets=(982 491 392 294 196 98 49 0)  
budget=${test_budgets[${SLURM_ARRAY_TASK_ID}]}
output_path="${OUTPUT_DIR}/${EXP_NAME}_${budget}"

#by sensor wearers
#sensor_wearers=9807
#wearers=${sensor_wearers[${SLURM_ARRAY_TASK_ID}]}
#output_path="${OUTPUT_DIR}/${EXP_NAME}_${wearers}}"

#parsed parameters 
wearers=9807
tested=0
network_size=1e4
I_min_threshold=0.0
I_max_threshold=1.0
user_fraction=1.0
batches_records=40
batches_tests=20
parflag=True
num_cores=16
learn_transition_rates=False
transition_rates_str='latent_periods,community_infection_periods'
transition_rates_noise='0.1,0.1'
learn_transimission_rate=True
transmission_rate_bias=0.0
transmission_rate_noise=0.25
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
  --assimilation-batches-perfect=${batches_records} \
  --assimilation-batches-imperfect=${batches_tests} \
  --parallel-flag=${parflag} \
  --num-cores=${num_cores} \
  --learn-transition-rates=${learn_transition_rates} \
  --transition-rates-str=${transition_rates_str} \
  --transition-rates-noise=${transition_rates_noise} \
  --learn-transmission-rate=${learn_transimission_rate} \
  --transmission-rate-bias=${transmission_rate_bias} \
  --transmission-rate-noise=${transmission_rate_noise} \
  >${stdout} 2>${stderr}




