#!/bin/sh
env="StarCraft2"
map="MMM2"
algo="rmappo"
exp="sanity_check"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=4 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 2 --episode_length 100 --num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --use_eval --add_center_xy --use_state_agent --gain 1 \
    --model_dir /home/eva_data/xyq
done

# mappo: --ppo_epoch 5 --clip_param 0.2 \
# mat: --ppo_epoch 15 --clip_param 0.05 \
d
# mappo
#/home/eva_data/xyq/on-policy/onpolicy/scripts/results/StarCraft2/6h_vs_8z/mappo/retrain/wandb/run-20220706_180611-1qogpf0r
# mat
#/home/eva_data/xyq/on-policy/onpolicy/scripts/results/StarCraft2/6h_vs_8z/mat/retrain/wandb/run-20220706_172156-3h1iua8u/files/transformer_3124.pt
# mat_dec
#/home/eva_data/xyq/on-policy/onpolicy/scripts/results/StarCraft2/6h_vs_8z/mat_dec/retrain/wandb/run-20220706_172139-2edigroz/files/transformer_3124.pt