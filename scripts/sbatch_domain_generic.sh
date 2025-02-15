#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 100G
#SBATCH --time 1-00:00:00
#SBATCH --gres gpu:1

python3 src/domain_generic.py \
    --domain_name cartpole \
    --task_name swingup \
    --action_repeat 8 \
    --mode train \
    --use_inv \
    --num_shared_layers 8 \
    --pad_checkpoint 500k \
    --seed 0 \
    --work_dir logs/cartpole_swingup \
    --save_dir logs/domain_generic/cartpole_swingup