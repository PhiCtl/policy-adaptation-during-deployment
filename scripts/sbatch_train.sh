#!/bin/bash
#SBATCH --exclude=i[20,24-25,27-28,36-40,53-57,65]
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 80G
#SBATCH --time 2-00:00:00
#SBATCH --gres gpu:1

python3 src/train.py \
    --domain_name walker \
    --task_name walk \
    --action_repeat 4 \
    --mode train \
    --predictor force_walker \
    --use_inv \
    --num_shared_layers 8 \
    --seed 0 \
    --work_dir logs/walker_walk/inv/0_-3 \
    --train_steps 4 \
    --force_walker 3.0