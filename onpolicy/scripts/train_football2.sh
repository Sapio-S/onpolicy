#!/bin/sh
# exp param
env="Football"
scenario="academy_pass_and_shoot_with_keeper"
algo="rmappo"
exp="env2_academy_pass_and_shoot_with_keeper"
seed=1

CUDA_VISIBLE_DEVICES=0 python train/train_football2.py \
--env_name ${env} --scenario_name ${scenario} \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} --clip_param 0.05 \
--num_agents 2 --representation simple115v2 --num_env_steps 100000000 --n_rollout_threads 50 --ppo_epoch 10 --episode_length 1000 --num_mini_batch 2 --data_chunk_length 10 --save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --eval_episodes 100 --n_eval_rollout_threads 100 --rewards scoring,checkpoints