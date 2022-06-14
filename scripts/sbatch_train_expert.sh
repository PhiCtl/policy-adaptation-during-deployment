#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 100G
#SBATCH --time 2-00:00:00
#SBATCH --gres gpu:1

python3 src/train.py \
    --domain_name cartpole \
    --task_name swingup \
    --action_repeat 8 \
    --mode train \
    --use_inv \
    --num_shared_layers 8 \
    --seed 0 \
    --pad_checkpoint 500k \
    --init_dir logs/cartpole_swingup/inv/0 \
    --work_dir logs/cartpole_swingup_0_3/inv/0 \
    --save_model \
    --cart_mass 0.3