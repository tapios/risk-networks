#!/bin/bash

##################################
# Infectiousness test experiment #
##################################
# set: fractions_test = % tested of I per day
# set: output_path = ${OUTPUT_DIR}/expt_name_${tested}
# ENSURE you have selected the parallel master equations import from _master_eqns_init.py
# submit with: sbatch run_infectious_test_expt_parallel.sh

# preface ######################################################################
set -euo pipefail

network_size=1e5
EXP_NAME="NYC_1e5"

tested=0
wearers=0
I_min_threshold=0.0
I_max_threshold=1.0
user_fraction=1.0
budget=10000
batches_sensor=1
batches_test=20
batches_records=400

num_cores=16

OUTPUT_DIR="output"
output_path="${OUTPUT_DIR}/${EXP_NAME}"
stdout="${output_path}/stdout"
stderr="${output_path}/stderr"

mkdir -p "${output_path}"

# launch #######################################################################
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
  --num-cores=${num_cores}
