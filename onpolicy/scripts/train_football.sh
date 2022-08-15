#!/bin/sh
env="football"
exp="0815"
seed_max=1

# scenario="academy_pass_and_shoot_with_keeper"
# n_agent=2
# episode_length=200

# scenario="academy_run_pass_and_shoot_with_keeper"
# n_agent=2
# episode_length=200

# scenario="academy_3_vs_1_with_keeper"
# n_agent=3
# episode_length=200

# scenario="academy_counterattack_easy"
# n_agent=4
# episode_length=200

scenario="academy_counterattack_hard"
n_agent=4
episode_length=1000

# scenario="academy_single_goal_versus_lazy"
# n_agent=11
# episode_length=1000

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    algo="rmappo"
    CUDA_VISIBLE_DEVICES=0 python train/train_football.py --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --n_agent ${n_agent} \
    --num_env_steps 25000000 --n_rollout_threads 50 --ppo_epoch 15 --episode_length  ${episode_length} --num_mini_batch 2 --data_chunk_length 10 --use_eval --eval_episodes 100 --n_eval_rollout_threads 100
    #--lr 5e-4 --entropy_coef 0.01 --max_grad_norm 0.5 --eval_episodes 32 --n_training_threads 16 --n_rollout_threads 20 --num_mini_batch 1 --episode_length 200 --eval_interval 100 --num_env_steps 10000000 --ppo_epoch 5 --clip_param 0.2 --layer_N 2 --use_eval --use_value_active_masks --use_policy_active_masks
    
    algo="mat"
    CUDA_VISIBLE_DEVICES=0 python train/train_football.py --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --n_agent ${n_agent} \
    --num_env_steps 25000000 --n_rollout_threads 50 --ppo_epoch 15 --episode_length  ${episode_length} --num_mini_batch 1 --use_eval --eval_episodes 100 --n_eval_rollout_threads 100 --ppo_epoch 10 --clip_param 0.05 --use_eval --use_value_active_masks --use_policy_active_masks --use_recurrent_policy
    # --lr 5e-4 --entropy_coef 0.01 --max_grad_norm 0.5 --eval_episodes 32 --n_training_threads 16 --n_rollout_threads 20 --num_mini_batch 1 --episode_length 200 --eval_interval 25 --num_env_steps 10000000 --ppo_epoch 10 --clip_param 0.05 --use_eval --use_value_active_masks --use_policy_active_masks --use_recurrent_policy

    algo="mat_dec"
    CUDA_VISIBLE_DEVICES=0 python train/train_football.py --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --n_agent ${n_agent} \
    --num_env_steps 25000000 --n_rollout_threads 50 --ppo_epoch 15 --episode_length  ${episode_length} --num_mini_batch 1 --use_eval --eval_episodes 100 --n_eval_rollout_threads 100 --ppo_epoch 10 --clip_param 0.05 --use_eval --use_value_active_masks --use_policy_active_masks --use_recurrent_policy
    # --lr 5e-4 --entropy_coef 0.01 --max_grad_norm 0.5 --eval_episodes 32 --n_training_threads 16 --n_rollout_threads 20 --num_mini_batch 1 --episode_length 200 --eval_interval 25 --num_env_steps 10000000 --ppo_epoch 10 --clip_param 0.05 --use_eval --use_value_active_masks --use_policy_active_masks --use_recurrent_policy

done