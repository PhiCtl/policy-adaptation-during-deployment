#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 110G
#SBATCH --time 2-30:00:00
#SBATCH --account vita
#SBATCH --gres gpu:1

python3 src/train_protta.py \
    --domain_name cartpole \
    --task_name swingup \
    --action_repeat 8 \
    --mode train \
    --dependent \
    --use_inv \
    --num_shared_layers 4 \
    --seed 0 \
    --work_dir logs/cartpole_swingup_protta/inv/0 \
    --save_model
