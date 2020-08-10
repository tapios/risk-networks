#!/bin/bash

#SBATCH --time=24:00:00                 # walltime
#SBATCH --ntasks=1                      # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=24G 
#SBATCH -J "Intervention_scenario"
#SBATCH --output=output/slurm_%j.out
#SBATCH --error=output/slurm_%j.err  
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-6

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
EXP_NAME="982_sens_intervention_time"

# parameters & constants #######################################################
#by sensor wearers
#sensor_wearers=(982)
#wearers=${sensor_wearers[${SLURM_ARRAY_TASK_ID}]}
#output_path="${OUTPUT_DIR}/${EXP_NAME}_${wearers}"

#by start time
intervention_start_times=(10 12 14 16 18 20 25)
start_time=${intervention_start_times[${SLURM_ARRAY_TASK_ID}]}
output_path="${OUTPUT_DIR}/${EXP_NAME}_${start_time}"

budget=49
tested=0
wearers=982
#parsed parameters 
network_size=1e3
I_min_threshold=0.0
I_max_threshold=1.0
user_fraction=1.0
batches=4
parflag=False
intervention_E_min_threshold=0.999
intervention_I_min_threshold=0.1
intervention_interval=1.0

stdout="${output_path}/stdout"
stderr="${output_path}/stderr"

mkdir -p "${output_path}"


# launch #######################################################################
python3 joint_epidemic_assimilation.py \
  --user-network-user-fraction=${user_fraction} \
  --constants-output-path=${output_path} \
  --observations-I-fraction-tested=${tested} \
  --observations-I-budget=${budget} \
  --observations-I-min-threshold=${I_min_threshold} \
  --observations-I-max-threshold=${I_max_threshold} \
  --observations-sensor-wearers=${wearers} \
  --network-node-count=${network_size} \
  --assimilation-batches=${batches} \
  --parallel-flag=${parflag} \
  --intervention-interval=${intervention_interval} \
  --intervention-E-min-threshold=${intervention_E_min_threshold} \
  --intervention-I-min-threshold=${intervention_I_min_threshold} \
  --intervention-start-time=${start_time} \
  >${stdout} 2>${stderr}


