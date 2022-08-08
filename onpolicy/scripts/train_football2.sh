#!/bin/sh
# exp param
env="Football"
scenario="academy_counterattack_easy"
algo="mat"
exp="check"
seed=1


# football param
num_agents=4
representation="simple115v2"

# train param
num_env_steps=25000000
episode_length=200

# log param
log_interval=200000
save_interval=200000

# eval param
eval_interval=400000
eval_episodes=100
n_eval_rollout_threads=100 

# tune param
n_rollout_threads=50
num_mini_batch=2


echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

CUDA_VISIBLE_DEVICES=2 python train/train_football2.py \
--env_name ${env} --scenario_name ${scenario} \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--num_agents ${num_agents} --representation ${representation} \
--num_env_steps ${num_env_steps} \
--n_rollout_threads ${n_rollout_threads} \
--episode_length ${episode_length} --num_mini_batch ${num_mini_batch} \
--save_interval ${save_interval} --log_interval ${log_interval} \
--use_eval \
--eval_interval ${eval_interval} --eval_episodes ${eval_episodes} \
--n_eval_rollout_threads ${n_eval_rollout_threads} \
--user_name "sapios" --rewards "scoring,checkpoints" \
--ppo_epoch 15 --clip_param 0.2 --use_recurrent_policy False