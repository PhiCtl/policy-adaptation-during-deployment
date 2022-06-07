#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 100G
#SBATCH --time 11:00:00
#SBATCH --gres gpu:1

python3 src/tta_il_agent.py \
    --domain_name walker \
    --task_name walk \
    --action_repeat 4 \
    --mode train \
    --use_inv \
    --num_shared_layers 8 \
    --pad_checkpoint 500k \
    --seed 0 \
    --train_steps 100000 \
    --work_dir logs/walker_walk \
    --save_dir logs/IL/shared/walker_walk 