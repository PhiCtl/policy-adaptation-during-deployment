#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 48G
#SBATCH --time 11:00:00
#SBATCH --account vita
#SBATCH --gres gpu:1

python3 src/imitation_learning.py \
    --domain_name cartpole \
    --task_name swingup \
    --action_repeat 8 \
    --mode train \
    --use_inv \
    --num_shared_layers 8 \
    --actor_lr 3e-4 \
    --encoder_lr 3e-4 \
    --ss_lr 3e-4 \
    --seed 0 \
    --work_dir logs/cartpole_swingup/inv/0 \
    --save_model