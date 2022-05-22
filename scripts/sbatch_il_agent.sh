#!/bin/bash
#SBATCH --exclude=i[20,24-25,27-28,36-40,53-57,65]
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 80G
#SBATCH --time 11:00:00
#SBATCH --gres gpu:1

python3 src/imitation_learning.py \
    --domain_name cartpole \
    --task_name swingup \
    --action_repeat 8 \
    --mode train \
    --use_inv \
    --num_shared_layers 8 \
    --pad_checkpoint 500k \
    --seed 0 \
    --work_dir logs/cartpole_swingup \
    --save_dir logs/IL/cartpole_swingup 