#!/bin/bash

#SBATCH --time=24:00:00                 # walltime
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G 
#SBATCH -J "Intervention_scenario"
#SBATCH --output=output/slurm_%A_%a.out
#SBATCH --error=output/slurm_%A_%a.err  
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-1

################################
# Intervention test experiment #
################################
# submit with: sbatch run_intervention_scenario_parallel.sh

# preface ######################################################################
set -euo pipefail

OUTPUT_DIR="output"
EXP_NAME="new_script"

# Experimental series parameters ###############################################
# intervention start time experiment
intervention_start_times=(10 20)
start_time=${intervention_start_times[${SLURM_ARRAY_TASK_ID}]}
output_path="${OUTPUT_DIR}/${EXP_NAME}_${start_time}"

# parameters & constants #######################################################

#network 
network_size=1e4

#user base
user_fraction=1.0

#testing: virus tests
I_min_threshold=0.0
I_max_threshold=1.0
budget=491
tested=0
batches_tests=1

#testing: sensor wearers
wearers=2451
batches_sensors=5

#testing: hospital/death records
batches_records=40

#intervention
int_freq='single'

#other
update_test="local"


#output parameters
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
  --assimilation-batches-sensor=${batches_sensors} \
  --assimilation-batches-test=${batches_tests} \
  --assimilation-batches-record=${batches_records} \
  --parallel-flag \
  --intervention-frequency=${int_freq} \
  --intervention-start-time=${start_time} \
  --assimilation-update-test=${update_test} \
  >${stdout} 2>${stderr}


