#!/bin/sh
env="football"
scenario="academy_pass_and_shoot_with_keeper"
# academy_pass_and_shoot_with_keeper
# academy_3_vs_1_with_keeper
# academy_counterattack_easy
n_agent=2
algo="mat_dec"
exp="0808"
seed_max=5

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=3 python train/train_football.py --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --n_agent ${n_agent} --lr 5e-4 --entropy_coef 0.01 --max_grad_norm 0.5 --eval_episodes 32 --n_training_threads 16 --n_rollout_threads 20 --num_mini_batch 1 --episode_length 200 --eval_interval 25 --num_env_steps 10000000 --ppo_epoch 10 --clip_param 0.05 --use_eval --use_value_active_masks --use_policy_active_masks --use_recurrent_policy
done