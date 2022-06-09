#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 48G
#SBATCH --time 07:30:00
#SBATCH --account vita
#SBATCH --gres gpu:1

python3 src/eval.py \
	--domain_name walker \
	--task_name walk \
	--action_repeat 4 \
	--mode train \
	--use_inv \
	--num_shared_layers 8 \
	--seed 0 \
	--work_dir logs/walker_walk/inv/0 \
	--pad_checkpoint 500k \
	--pad_num_episodes 100 \
	--episode_length 1000