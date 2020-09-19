#!/bin/bash

#SBATCH --time=120:00:00                 # walltime
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=386G
#SBATCH -J "Intervention_scenario"
#SBATCH --output=output/slurm_%A_%a.out
#SBATCH --error=output/slurm_%A_%a.err  
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-0

################################
# Intervention test experiment #
################################
# submit with: sbatch run_intervention_scenario_parallel.sh

# preface ######################################################################
set -euo pipefail

# parallelization
num_cpus=${SLURM_CPUS_PER_TASK}
bytes_of_memory=$((${SLURM_MEM_PER_NODE}*1000000 / 8)) #MB -> bytes
echo "requested ${num_cpus} cores and ray is told ${bytes_of_memory} memory available"
# parameters & constants #######################################################

# network  + sensor wearers

#EXP_NAME="NYC_1e3"
#network_size=1e3
#wearers=245
#batches_sensors=5
#batches_records=4
#budget=491
#batches_tests=1
#tested=0

EXP_NAME="noDA_NYC_1e4"
network_size=1e4
wearers=2451
batches_sensors=5
batches_records=40
budget=491
batches_tests=1
tested=0

# EXP_NAME="noDA_NYC_1e5"
# network_size=1e5
# wearers=0
# batches_sensors=1
# batches_records=400
# budget=25000
# batches_tests=50
# tested=0


# user base
user_fraction=1.0

# testing: virus tests
I_min_threshold=0.0
I_max_threshold=1.0

# intervention
int_freq='single'

# other
update_test="local"

# Experimental series parameters ###############################################
# intervention start time experiment
intervention_start_times=18
start_time=${intervention_start_times[${SLURM_ARRAY_TASK_ID}]}


# output parameters
OUTPUT_DIR="output"
output_path="${OUTPUT_DIR}/${EXP_NAME}_${start_time}"
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
  --parallel-memory=${bytes_of_memory} \
  --parallel-num-cpus=${num_cpus} \
  --intervention-frequency=${int_freq} \
  --intervention-start-time=${start_time} \
  --assimilation-update-test=${update_test} \
  >${stdout} 2>${stderr}


