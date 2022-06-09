#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 100G
#SBATCH --time 1-00:00:00
#SBATCH --gres gpu:1

python3 src/domain_generic.py \
    --domain_name walker \
    --task_name walk \
    --action_repeat 4 \
    --mode train \
    --use_inv \
    --num_shared_layers 8 \
    --pad_checkpoint 1000k \
    --seed 0 \
    --work_dir logs/walker_walk \
    --save_dir logs/domain_generic/walker_walk 