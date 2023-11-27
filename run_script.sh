#!/bin/bash -l

config_file=./config/AFNO.yaml
config='afno_backbone'
run_num='mass'

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(hostname)

set -x
source export_DDP_vars.sh
python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num

