#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 48G
#SBATCH --time 02:30:00
#SBATCH --account vita
#SBATCH --gres gpu:1

python3 src/eval.py \
	--domain_name cartpole \
	--task_name swingup \
	--action_repeat 8 \
	--mode steady \
	--use_inv \
	--num_shared_layers 8 \
	--seed 0 \
	--work_dir logs/cartpole_swingup/inv/0 \
	--pad_checkpoint 500k \
	--pad_num_episodes 1 \
	--episode_length 3000 \
	--dependent \
	--threshold 3 \
	--save_video

