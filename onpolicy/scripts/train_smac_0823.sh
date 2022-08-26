#!/bin/sh
env="StarCraft2"
map="MMM2"
exp="0823_episode_length100"
seed_max=1

echo "env is ${env}, map is ${map}, exp is ${exp}, max seed is ${seed_max}"
for seed in {3..5..1}
do
    echo "seed is ${seed}:"

    algo="rmappo"
    echo "algo is ${algo}"
    CUDA_VISIBLE_DEVICES=0 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 32 --episode_length 100 --num_env_steps 10000000 \
    --num_mini_batch 2 --ppo_epoch 5 --gain 1 --use_value_active_masks --use_eval --layer_N 1 #--use_recurrent_policy
done

# # !/bin/sh
# env="StarCraft2"
# map="6h_vs_8z"
# exp="0823_episode_length400"
# seed_max=1

# echo "env is ${env}, map is ${map}, exp is ${exp}, max seed is ${seed_max}"

# for seed in {5..5..1}
# do
#     echo "seed is ${seed}:"
#     algo="mappo"
#     echo "algo is ${algo}"
#     CUDA_VISIBLE_DEVICES=1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
#     --n_training_threads 1 --n_rollout_threads 8 --episode_length 400 --num_env_steps 10000000 \
#     --num_mini_batch 1 --ppo_epoch 5 --gain 0.01 --use_value_active_masks --use_eval --use_recurrent_policy
# done
